#![allow(missing_docs)]

//! Benchmarks for reading the best transactions iterator from a large pending pool.
//!
//! Payload builders take a fresh
//! [`BestTransactions`](reth_transaction_pool::pool::BestTransactions) snapshot for every payload
//! they build, while the pool's read lock is held, and usually consume only a small prefix of it
//! before the payload is full or its deadline passes.

use alloy_primitives::{Address, TxHash, U256};
use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use reth_primitives_traits::SealedBlock;
use reth_transaction_pool::{
    blobstore::InMemoryBlobStore,
    noop::MockTransactionValidator,
    pool::PoolInner,
    test_utils::{MockOrdering, MockTransaction},
    validate::ValidTransaction,
    CanonicalStateUpdate, PoolConfig, PoolTransaction, PoolUpdateKind, SubPoolLimit,
    TransactionOrigin, TransactionValidationOutcome,
};
use std::hint::black_box;

type BenchPool =
    PoolInner<MockTransactionValidator<MockTransaction>, MockOrdering, InMemoryBlobStore>;

const BASE_FEE: u64 = 100;

/// Transactions a builder typically consumes from one snapshot before it stops.
const TAKE: usize = 512;

fn sender_address(index: usize) -> Address {
    let mut bytes = [0u8; 20];
    bytes[12..].copy_from_slice(&(index as u64 + 1).to_be_bytes());
    Address::from(bytes)
}

type Batch = Vec<(TransactionOrigin, TransactionValidationOutcome<MockTransaction>)>;

/// Builds `txs_per_sender` gapless pending transactions for each sender in `senders`.
fn pending_batch(senders: std::ops::Range<usize>, txs_per_sender: u64) -> Batch {
    let mut batch = Vec::with_capacity(senders.len() * txs_per_sender as usize);
    for s in senders {
        let sender = sender_address(s);
        let fee = BASE_FEE as u128 + (s % 1_000) as u128;
        for nonce in 0..txs_per_sender {
            let mut tx = MockTransaction::eip1559();
            tx.set_sender(sender)
                .set_nonce(nonce)
                .set_max_fee(fee)
                .set_priority_fee(1 + (s % 997) as u128)
                .set_gas_limit(100_000)
                .set_size(640);
            batch.push((
                TransactionOrigin::External,
                TransactionValidationOutcome::Valid {
                    balance: U256::MAX,
                    state_nonce: 0,
                    bytecode_hash: None,
                    transaction: ValidTransaction::Valid(tx),
                    propagate: false,
                    authorities: None,
                },
            ));
        }
    }
    batch
}

/// Builds an uncapped pending pool of `senders` senders with `txs_per_sender` gapless
/// transactions each, with spread fees so ordering is non-trivial.
fn build_pending_pool(senders: usize, txs_per_sender: u64) -> BenchPool {
    let pool = PoolInner::new(
        MockTransactionValidator::default(),
        MockOrdering::default(),
        InMemoryBlobStore::default(),
        PoolConfig {
            pending_limit: SubPoolLimit::max(),
            max_account_slots: txs_per_sender as usize,
            ..Default::default()
        },
    );

    let mut block = reth_ethereum_primitives::Block::default();
    block.header.gas_limit = 30_000_000;
    block.header.base_fee_per_gas = Some(BASE_FEE);
    let block = SealedBlock::seal_slow(block);
    pool.on_canonical_state_change(CanonicalStateUpdate {
        new_tip: &block,
        pending_block_base_fee: BASE_FEE,
        pending_block_blob_fee: None,
        changed_accounts: Vec::new(),
        mined_transactions: Vec::new(),
        update_kind: PoolUpdateKind::Commit,
    });

    let batch = pending_batch(0..senders, txs_per_sender);
    pool.add_transactions_with_origins(batch);
    assert_eq!(pool.size().pending, senders * txs_per_sender as usize);
    pool
}

fn bench_best_transactions(c: &mut Criterion) {
    let mut group = c.benchmark_group("best_transactions");
    group.sample_size(30);

    for (senders, txs_per_sender) in [(1_000, 1), (10_000, 1), (40_000, 1), (10_000, 4)] {
        let pool = build_pending_pool(senders, txs_per_sender);
        let shape = format!("senders={senders}/txs_per_sender={txs_per_sender}");

        // Snapshot plus drop: the cost a builder pays before it reads the first transaction.
        group.bench_function(BenchmarkId::new("snapshot", &shape), |b| {
            b.iter(|| black_box(pool.best_transactions()))
        });

        group.bench_function(BenchmarkId::new(format!("take_{TAKE}"), &shape), |b| {
            b.iter(|| {
                let mut best = pool.best_transactions();
                for _ in 0..TAKE {
                    black_box(best.next());
                }
            })
        });

        if senders * txs_per_sender as usize <= 40_000 {
            group.bench_function(BenchmarkId::new("drain", &shape), |b| {
                b.iter(|| {
                    for tx in pool.best_transactions() {
                        black_box(tx);
                    }
                })
            });
        }
    }

    group.finish();
}

/// Benchmarks inserting new senders into a large pending pool while a builder holds a snapshot of
/// it, so the pool's ordered structures are shared with the live iterator.
fn bench_add_with_live_snapshot(c: &mut Criterion) {
    let mut group = c.benchmark_group("best_transactions");
    group.sample_size(30);

    const SENDERS: usize = 40_000;
    const BATCH: usize = 128;
    let pool = build_pending_pool(SENDERS, 1);
    let mut added: Vec<TxHash> = Vec::new();
    let mut next_sender = SENDERS;

    group.bench_function(
        BenchmarkId::new(format!("add_{BATCH}_with_live_snapshot"), format!("senders={SENDERS}")),
        |b| {
            b.iter_batched(
                || {
                    // keep the pool at a constant size: drop the previous batch (untimed)
                    pool.remove_transactions(std::mem::take(&mut added));
                    let batch = pending_batch(next_sender..next_sender + BATCH, 1);
                    next_sender += BATCH;
                    added = batch
                        .iter()
                        .map(|(_, outcome)| match outcome {
                            TransactionValidationOutcome::Valid { transaction, .. } => {
                                *transaction.transaction().hash()
                            }
                            _ => unreachable!(),
                        })
                        .collect();
                    (pool.best_transactions(), batch)
                },
                |(snapshot, batch)| {
                    pool.add_transactions_with_origins(batch);
                    snapshot
                },
                BatchSize::PerIteration,
            )
        },
    );

    group.finish();
}

criterion_group!(benches, bench_best_transactions, bench_add_with_live_snapshot);
criterion_main!(benches);
