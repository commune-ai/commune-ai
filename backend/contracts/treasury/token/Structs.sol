// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts v4.4.1 (token/ERC721/ERC721.sol)

pragma solidity ^0.8.0;

struct UserState {
    address user;
    uint256[] assets;
    uint256[] balances;
    uint256 lastUpdateBlock;
}

struct AssetState {
    string name;
    uint256 balance;
    uint256 value;
    address asset;
    bytes valueOracle;
    // options ERC20, ERC721, 
    string type;
    bytes metaData;
}

