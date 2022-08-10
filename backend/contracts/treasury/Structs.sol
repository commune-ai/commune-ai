// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts v4.4.1 (token/ERC721/ERC721.sol)

pragma solidity ^0.8.0;

struct UserState {
    address user;
    address[] assets;
    uint256 lastUpdateBlock;
}


struct AssetState {
    address asset;
    string name;
    string mode;
    uint256 balance;
    uint256 value;
    address valueOracle;
    bool liquid;
    uint256 lastUpdateBlock;
    bytes metaData;
}

