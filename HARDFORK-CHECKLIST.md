# Checklist for Integrating New Hard Fork / Devnet Changes

## Introducing New EIP Types or Changes to Primitive Types

- [ ] Implement required changes to primitive data structures in [alloy](https://github.com/alloy-rs/alloy).
- [ ] Ensure all new EIP data structures, constants, and helpers are first added to the `alloy-eips` crate.
- [ ] Add new transaction types to `alloy-consensus`.
- [ ] If existing data structures like `Header` or `Block` are modified, apply updates to `alloy-consensus` (e.g., adding the `requests_hash` field for the Prague hard fork).
- [ ] **Security Check:** Verify that RLP/SSZ serialization for all new types is strictly tested against malleability attacks.

## Engine API Updates

- [ ] Add new types to the `alloy-rpc-types-engine` crate for Engine API changes (e.g., new `engine_newPayloadV3/V4` and `engine_getPayloadV3/V4` pairs).
- [ ] If `engine_newPayloadVX` has new parameters, update the `ExecutionPayloadSidecar` container type.
- [ ] **Validation:** Ensure `ExecutionPayloadSidecar` fields are correctly mapped to prevent node crashes during block conversion.

## Reth Implementation

### Updates to the Engine API
- [ ] Add new endpoints to the `EngineApi` trait and provide full implementations.
- [ ] Update conversion logic for `ExecutionPayload` + `ExecutionPayloadSidecar` to `Block` for any new parameters.
- [ ] Update version-specific validation logic within the `EngineValidator` trait.

## Op-Reth Specific Changes

### Updates to the Engine API
- [ ] Op-stack follows L1 Engine API closely. For deviations (like the additional fields in Isthmus), implement dedicated server traits in `OpEngineApi`.
- [ ] Mirror L1 versioned endpoint changes for dedicated OP types.

### Hardfork Management
- [ ] Map dedicated Op-stack hardforks (e.g., Holocene, Isthmus) to their L1 equivalents in the `ChainSpec`.
- [ ] **Mapping Example:** `OpHardfork::Isthmus` must correspond to `EthereumHardfork::Prague`.
- [ ] **Critical Security Check:** Ensure that the activation timestamp/block for `OpHardfork` is correctly synchronized with the L1 `EthereumHardfork` to prevent state root divergence and consensus splits.
- [ ] Define these mappings explicitly within the `ChainSpec` to enforce consistency during synchronization.
