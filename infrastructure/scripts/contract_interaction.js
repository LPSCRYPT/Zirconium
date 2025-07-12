const { ethers } = require("hardhat");
const fs = require("fs");
const path = require("path");

// Load contract addresses
const sepoliaAddresses = JSON.parse(
  fs.readFileSync(path.join(__dirname, "../config/sepolia-addresses.json"), "utf8")
);

async function main() {
  console.log("🚀 Zirconium Contract Interaction Demo");
  console.log("=" * 50);

  // Get signer
  const [signer] = await ethers.getSigners();
  console.log(`📝 Using account: ${signer.address}`);
  
  // Get balance
  const balance = await signer.getBalance();
  console.log(`💰 Balance: ${ethers.utils.formatEther(balance)} ETH`);

  if (balance.lt(ethers.utils.parseEther("0.01"))) {
    console.log("⚠️  Warning: Low balance, may not be able to send transactions");
  }

  // Load contract ABIs and connect to deployed contracts
  const contracts = await loadContracts(signer);

  console.log("\n🔍 Contract Information:");
  console.log("-".repeat(30));
  
  for (const [name, contract] of Object.entries(contracts)) {
    console.log(`${name}: ${contract.address}`);
    
    // Get some basic info from each contract
    try {
      if (name.includes("Verifier")) {
        const archInfo = await contract.getArchitectureInfo();
        console.log(`  Architecture: ${archInfo.architecture}`);
        console.log(`  Circuit Gates: ${archInfo.circuitGates.toString()}`);
        console.log(`  Security Level: ${archInfo.securityLevel.toString()}-bit`);
      }
    } catch (error) {
      console.log(`  Error reading info: ${error.message}`);
    }
  }

  // Demo interactions
  await demoVerifierInteraction(contracts);
  await demoProofGeneration(contracts);
}

async function loadContracts(signer) {
  const contracts = {};

  try {
    // Load RWKV Verifier
    const RWKVVerifier = await ethers.getContractFactory("ProductionRWKVVerifier");
    contracts.RWKVVerifier = RWKVVerifier.attach(sepoliaAddresses.contracts.ProductionRWKVVerifier).connect(signer);

    // Load Mamba Verifier
    const MambaVerifier = await ethers.getContractFactory("ProductionMambaVerifier");
    contracts.MambaVerifier = MambaVerifier.attach(sepoliaAddresses.contracts.ProductionMambaVerifier).connect(signer);

    // Load xLSTM Verifier
    const xLSTMVerifier = await ethers.getContractFactory("ProductionxLSTMVerifier");
    contracts.xLSTMVerifier = xLSTMVerifier.attach(sepoliaAddresses.contracts.ProductionxLSTMVerifier).connect(signer);

    console.log("✅ Successfully connected to all verifier contracts");
    return contracts;

  } catch (error) {
    console.error("❌ Error loading contracts:", error.message);
    process.exit(1);
  }
}

async function demoVerifierInteraction(contracts) {
  console.log("\n🧪 Demo: Verifier Contract Interaction");
  console.log("-".repeat(40));

  try {
    // Get verification key from RWKV verifier
    const vkInfo = await contracts.RWKVVerifier.getVerificationKey();
    console.log("📋 RWKV Verification Key Info:");
    console.log(`  Alpha: [${vkInfo.alpha[0].toString()}, ${vkInfo.alpha[1].toString()}]`);
    console.log(`  IC Length: ${vkInfo.icLength.toString()}`);

    // Get gas estimate for verification
    const gasEstimate = await contracts.RWKVVerifier.estimateVerificationGas(4);
    console.log(`⛽ Estimated gas for verification: ${gasEstimate.toString()}`);

    // Get user stats (should be 0 for new address)
    const userStats = await contracts.RWKVVerifier.getUserStats(await contracts.RWKVVerifier.signer.getAddress());
    console.log(`📊 User verification count: ${userStats.toString()}`);

  } catch (error) {
    console.error("❌ Error in verifier interaction:", error.message);
  }
}

async function demoProofGeneration(contracts) {
  console.log("\n🔐 Demo: Mock Proof Generation and Verification");
  console.log("-".repeat(50));

  // Generate mock proof data (in real usage, this would come from EZKL)
  const mockProof = generateMockProof();
  const mockPublicInputs = generateMockPublicInputs();

  console.log("📦 Generated mock proof data:");
  console.log(`  Proof points: ${Object.keys(mockProof).length} components`);
  console.log(`  Public inputs: ${mockPublicInputs.length} values`);

  try {
    // Estimate gas for proof verification
    console.log("\n⛽ Estimating gas for proof verification...");
    
    // Note: This will likely fail because our mock proof won't pass verification
    // But it demonstrates the interaction pattern
    const gasEstimate = await contracts.RWKVVerifier.estimateGas.verifyProof(mockProof, mockPublicInputs);
    console.log(`Gas estimate: ${gasEstimate.toString()}`);

    // In a real scenario, you would:
    // 1. Generate actual EZKL proof from neural network
    // 2. Call verifyProof() with real proof data
    // 3. Handle the verification result
    
    console.log("\n📝 To use with real proofs:");
    console.log("1. Generate EZKL proof from neural network computation");
    console.log("2. Call contract.verifyProof(realProof, realPublicInputs)");
    console.log("3. Check verification result and handle receipt");

  } catch (error) {
    console.log(`⚠️  Expected error with mock proof: ${error.message}`);
    console.log("   (This is normal - mock proofs don't pass cryptographic verification)");
  }
}

function generateMockProof() {
  // Generate mock proof structure (same format as contract expects)
  return {
    a: [
      ethers.BigNumber.from("0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"),
      ethers.BigNumber.from("0xfedcba0987654321fedcba0987654321fedcba0987654321fedcba0987654321")
    ],
    b: [
      [
        ethers.BigNumber.from("0x1111111111111111111111111111111111111111111111111111111111111111"),
        ethers.BigNumber.from("0x2222222222222222222222222222222222222222222222222222222222222222")
      ],
      [
        ethers.BigNumber.from("0x3333333333333333333333333333333333333333333333333333333333333333"),
        ethers.BigNumber.from("0x4444444444444444444444444444444444444444444444444444444444444444")
      ]
    ],
    c: [
      ethers.BigNumber.from("0x5555555555555555555555555555555555555555555555555555555555555555"),
      ethers.BigNumber.from("0x6666666666666666666666666666666666666666666666666666666666666666")
    ],
    z: [
      ethers.BigNumber.from("0x7777777777777777777777777777777777777777777777777777777777777777"),
      ethers.BigNumber.from("0x8888888888888888888888888888888888888888888888888888888888888888")
    ],
    t1: [
      ethers.BigNumber.from("0x9999999999999999999999999999999999999999999999999999999999999999"),
      ethers.BigNumber.from("0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    ],
    t2: [
      ethers.BigNumber.from("0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"),
      ethers.BigNumber.from("0xcccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc")
    ],
    t3: [
      ethers.BigNumber.from("0xdddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"),
      ethers.BigNumber.from("0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
    ],
    eval_a: ethers.BigNumber.from("0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"),
    eval_b: ethers.BigNumber.from("0x1111111111111111111111111111111111111111111111111111111111111110"),
    eval_c: ethers.BigNumber.from("0x1111111111111111111111111111111111111111111111111111111111111111"),
    eval_s1: ethers.BigNumber.from("0x1111111111111111111111111111111111111111111111111111111111111112"),
    eval_s2: ethers.BigNumber.from("0x1111111111111111111111111111111111111111111111111111111111111113"),
    eval_zw: ethers.BigNumber.from("0x1111111111111111111111111111111111111111111111111111111111111114")
  };
}

function generateMockPublicInputs() {
  // Generate 4 mock public inputs (required by the contract)
  return [
    ethers.BigNumber.from("12345"),
    ethers.BigNumber.from("67890"),
    ethers.BigNumber.from("13579"),
    ethers.BigNumber.from("24680")
  ];
}

async function demoAdvancedInteractions(contracts) {
  console.log("\n🔗 Demo: Advanced Contract Features");
  console.log("-".repeat(40));

  try {
    // Demo batch verification preparation
    console.log("📦 Preparing batch verification demo...");
    
    const mockProofs = [generateMockProof(), generateMockProof()];
    const mockInputs = [generateMockPublicInputs(), generateMockPublicInputs()];

    console.log(`Generated ${mockProofs.length} mock proofs for batch verification`);
    
    // In real usage, you would call batchVerifyProofs here
    console.log("📝 To use batch verification:");
    console.log("   contract.batchVerifyProofs(realProofs, realInputs)");

    // Demo model inference tracking
    console.log("\n📊 Model inference tracking example:");
    console.log("   1. Generate proof from actual model inference");
    console.log("   2. Call verifyModelInference() with prompt and continuation");
    console.log("   3. Receive inference ID for tracking");

  } catch (error) {
    console.error("❌ Error in advanced interactions:", error.message);
  }
}

// Error handling wrapper
async function runDemo() {
  try {
    await main();
    console.log("\n✅ Demo completed successfully!");
  } catch (error) {
    console.error("\n❌ Demo failed:", error.message);
    if (error.stack) {
      console.error("Stack trace:", error.stack);
    }
    process.exit(1);
  }
}

// Run the demo
if (require.main === module) {
  runDemo();
}

module.exports = {
  loadContracts,
  generateMockProof,
  generateMockPublicInputs
};