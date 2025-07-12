require("@nomicfoundation/hardhat-toolbox");
require("hardhat-gas-reporter");
require('dotenv').config();

module.exports = {
  solidity: {
    compilers: [
      {
        version: "0.8.19",
        settings: {
          optimizer: {
            enabled: true,
            runs: 1,  // Aggressive optimization for large contracts
            details: {
              yul: true
            }
          }
        }
      },
      {
        version: "0.8.24",
        settings: {
          optimizer: {
            enabled: true,
            runs: 1,  // Aggressive optimization for large contracts
            details: {
              yul: true
            }
          }
        }
      }
    ]
  },
  networks: {
    localhost: {
      url: "http://127.0.0.1:8545",
      allowUnlimitedContractSize: true,
      blockGasLimit: 30000000
    },
    hardhat: {
      chainId: 31337,
      allowUnlimitedContractSize: true,
      blockGasLimit: 30000000
    },
    sepolia: {
      url: process.env.SEPOLIA_URL || "https://eth-sepolia.g.alchemy.com/v2/demo",
      accounts: process.env.PRIVATE_KEY ? [process.env.PRIVATE_KEY] : [],
      chainId: 11155111,
      gasPrice: 20000000000
    },
    goerli: {
      url: process.env.GOERLI_URL || "https://eth-goerli.g.alchemy.com/v2/demo",
      accounts: process.env.PRIVATE_KEY ? [process.env.PRIVATE_KEY] : [],
      chainId: 5,
      gasPrice: 20000000000
    },
    polygon_mumbai: {
      url: process.env.POLYGON_MUMBAI_URL || "https://rpc-mumbai.maticvigil.com/",
      accounts: process.env.PRIVATE_KEY ? [process.env.PRIVATE_KEY] : [],
      chainId: 80001,
      gasPrice: 20000000000
    },
    mainnet: {
      url: process.env.MAINNET_URL || "",
      accounts: process.env.PRIVATE_KEY ? [process.env.PRIVATE_KEY] : []
    },
    flow_testnet: {
      url: process.env.FLOW_TESTNET_URL || "https://testnet.evm.nodes.onflow.org",
      accounts: process.env.PRIVATE_KEY ? [process.env.PRIVATE_KEY] : [],
      chainId: 545,
      gasPrice: 20000000000
    },
    flow_mainnet: {
      url: process.env.FLOW_MAINNET_URL || "https://mainnet.evm.nodes.onflow.org",
      accounts: process.env.PRIVATE_KEY ? [process.env.PRIVATE_KEY] : [],
      chainId: 747,
      gasPrice: 20000000000
    },
    zircuit_testnet: {
      url: process.env.ZIRCUIT_TESTNET_URL || "https://garfield-testnet.zircuit.com",
      accounts: process.env.PRIVATE_KEY ? [process.env.PRIVATE_KEY] : [],
      chainId: 48898,
      gasPrice: 10000000000
    },
    zircuit_mainnet: {
      url: process.env.ZIRCUIT_MAINNET_URL || "https://mainnet.zircuit.com",
      accounts: process.env.PRIVATE_KEY ? [process.env.PRIVATE_KEY] : [],
      chainId: 48900,
      gasPrice: 20000000000
    }
  },
  gasReporter: {
    enabled: process.env.REPORT_GAS !== undefined,
    currency: 'USD',
    gasPrice: 20,
    showTimeSpent: true
  },
  etherscan: {
    apiKey: process.env.ETHERSCAN_API_KEY
  }
};
