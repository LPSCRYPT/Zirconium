const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("🔗 Minimal EZKL Proof-Chaining", function () {
  let rwkvVerifier, mambaVerifier, xlstmVerifier, orchestrator;
  let owner, user1, user2;
  
  // Mock proof data (simulating EZKL-generated proofs)
  const mockProof1 = ethers.hexlify(ethers.randomBytes(64)); // 64 bytes mock proof
  const mockProof2 = ethers.hexlify(ethers.randomBytes(64));
  const mockProof3 = ethers.hexlify(ethers.randomBytes(64));
  
  const mockInputs = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000];

  beforeEach(async function () {
    console.log("  🚀 Setting up test environment...");
    
    // Get signers
    [owner, user1, user2] = await ethers.getSigners();
    
    // Deploy wrapper verifiers (with inheritance structure)
    const RWKVVerifier = await ethers.getContractFactory("RWKVVerifierWrapper");
    rwkvVerifier = await RWKVVerifier.deploy();
    await rwkvVerifier.waitForDeployment();
    
    const MambaVerifier = await ethers.getContractFactory("MambaVerifierWrapper");
    mambaVerifier = await MambaVerifier.deploy();
    await mambaVerifier.waitForDeployment();
    
    const xLSTMVerifier = await ethers.getContractFactory("xLSTMVerifierWrapper");
    xlstmVerifier = await xLSTMVerifier.deploy();
    await xlstmVerifier.waitForDeployment();
    
    // Deploy orchestrator
    const ProofChainOrchestrator = await ethers.getContractFactory("ProofChainOrchestrator");
    orchestrator = await ProofChainOrchestrator.deploy();
    await orchestrator.waitForDeployment();
    
    console.log("  ✅ Test environment ready");
  });

  describe("Individual Verifier Tests", function () {
    
    it("🏗️ Should deploy all verifiers with correct architectures", async function () {
      expect(await rwkvVerifier.getArchitecture()).to.equal("RWKV");
      expect(await mambaVerifier.getArchitecture()).to.equal("Mamba");
      expect(await xlstmVerifier.getArchitecture()).to.equal("xLSTM");
    });

    it("📊 Should have zero initial verifications", async function () {
      expect(await rwkvVerifier.getTotalVerifications()).to.equal(0);
      expect(await mambaVerifier.getTotalVerifications()).to.equal(0);
      expect(await xlstmVerifier.getTotalVerifications()).to.equal(0);
    });

    it("✅ Should verify proofs and emit events", async function () {
      // Test RWKV verification
      await expect(rwkvVerifier.verify(mockProof1, mockInputs))
        .to.emit(rwkvVerifier, "ProofVerified")
        .withArgs(ethers.keccak256(mockProof1), owner.address, true);
      
      expect(await rwkvVerifier.getTotalVerifications()).to.equal(1);
    });

    it("🚫 Should prevent proof replay attacks", async function () {
      // First verification should succeed
      await rwkvVerifier.verify(mockProof1, mockInputs);
      
      // Second verification with same proof should fail
      await expect(rwkvVerifier.verify(mockProof1, mockInputs))
        .to.be.revertedWith("Proof already used");
    });

    it("❌ Should revert on empty proofs or inputs", async function () {
      await expect(rwkvVerifier.verify("0x", mockInputs))
        .to.be.revertedWith("Empty proof");
      
      await expect(rwkvVerifier.verify(mockProof1, []))
        .to.be.revertedWith("No public inputs");
    });

    it("🔍 Should track proof verification status", async function () {
      const proofHash = ethers.keccak256(mockProof1);
      
      // Initially not verified
      expect(await rwkvVerifier.isProofVerified(proofHash)).to.be.false;
      
      // After verification
      await rwkvVerifier.verify(mockProof1, mockInputs);
      expect(await rwkvVerifier.isProofVerified(proofHash)).to.be.true;
    });
  });

  describe("Proof-Chaining Tests", function () {
    
    it("🎭 Should execute simple proof chain successfully", async function () {
      const verifiers = [
        await rwkvVerifier.getAddress(),
        await mambaVerifier.getAddress(),
        await xlstmVerifier.getAddress()
      ];
      
      const proofs = [mockProof1, mockProof2, mockProof3];
      
      const tx = await orchestrator.executeChain(verifiers, proofs, mockInputs);
      const receipt = await tx.wait();
      
      // Check for events
      const chainExecutedEvent = receipt.logs.find(
        log => log.fragment && log.fragment.name === "ChainExecuted"
      );
      expect(chainExecutedEvent).to.not.be.undefined;
      
      // Check execution was successful
      const executionId = chainExecutedEvent.args[0];
      const [success, stepsCompleted, finalOutputs] = await orchestrator.getChainResult(executionId);
      
      expect(success).to.be.true;
      expect(stepsCompleted).to.equal(3);
      expect(finalOutputs.length).to.equal(10);
    });

    it("📈 Should increment total executions", async function () {
      const initialCount = await orchestrator.totalExecutions();
      
      const verifiers = [await rwkvVerifier.getAddress()];
      const proofs = [mockProof1];
      
      await orchestrator.executeChain(verifiers, proofs, mockInputs);
      
      expect(await orchestrator.totalExecutions()).to.equal(initialCount + 1n);
    });

    it("🎯 Should emit ChainStepCompleted events", async function () {
      const verifiers = [
        await rwkvVerifier.getAddress(),
        await mambaVerifier.getAddress()
      ];
      
      const proofs = [mockProof1, mockProof2];
      
      await expect(orchestrator.executeChain(verifiers, proofs, mockInputs))
        .to.emit(orchestrator, "ChainStepCompleted");
    });

    it("❌ Should handle failed verifications gracefully", async function () {
      // Use same proof twice to trigger replay protection
      await rwkvVerifier.verify(mockProof1, mockInputs);
      
      const verifiers = [await rwkvVerifier.getAddress()];
      const proofs = [mockProof1]; // This will fail due to replay
      
      const tx = await orchestrator.executeChain(verifiers, proofs, mockInputs);
      const receipt = await tx.wait();
      
      const chainExecutedEvent = receipt.logs.find(
        log => log.fragment && log.fragment.name === "ChainExecuted"
      );
      
      const executionId = chainExecutedEvent.args[0];
      const [success, stepsCompleted] = await orchestrator.getChainResult(executionId);
      
      expect(success).to.be.false;
      expect(stepsCompleted).to.equal(0);
    });

    it("🔧 Should validate input parameters", async function () {
      // Empty verifiers array
      await expect(orchestrator.executeChain([], [], mockInputs))
        .to.be.revertedWith("No verifiers");
      
      // Mismatched arrays
      const verifiers = [await rwkvVerifier.getAddress()];
      const proofs = [mockProof1, mockProof2]; // Too many proofs
      
      await expect(orchestrator.executeChain(verifiers, proofs, mockInputs))
        .to.be.revertedWith("Length mismatch");
    });

    it("🔍 Should provide execution details", async function () {
      const verifiers = [await rwkvVerifier.getAddress()];
      const proofs = [mockProof1];
      
      const tx = await orchestrator.executeChain(verifiers, proofs, mockInputs);
      const receipt = await tx.wait();
      
      const chainExecutedEvent = receipt.logs.find(
        log => log.fragment && log.fragment.name === "ChainExecuted"
      );
      const executionId = chainExecutedEvent.args[0];
      
      const [executor, timestamp, success, stepsCompleted] = await orchestrator.getExecutionDetails(executionId);
      
      expect(executor).to.equal(owner.address);
      expect(success).to.be.true;
      expect(stepsCompleted).to.equal(1);
      expect(timestamp).to.be.greaterThan(0);
    });

    it("✅ Should confirm execution exists", async function () {
      const verifiers = [await rwkvVerifier.getAddress()];
      const proofs = [mockProof1];
      
      const tx = await orchestrator.executeChain(verifiers, proofs, mockInputs);
      const receipt = await tx.wait();
      
      const chainExecutedEvent = receipt.logs.find(
        log => log.fragment && log.fragment.name === "ChainExecuted"
      );
      const executionId = chainExecutedEvent.args[0];
      
      expect(await orchestrator.executionExists(executionId)).to.be.true;
      
      // Random execution ID should not exist
      const randomId = ethers.keccak256(ethers.toUtf8Bytes("random"));
      expect(await orchestrator.executionExists(randomId)).to.be.false;
    });
  });

  describe("Integration Tests", function () {
    
    it("🌊 Should handle multiple users executing chains", async function () {
      const verifiers = [await rwkvVerifier.getAddress()];
      
      // User1 executes chain
      const proof1 = ethers.hexlify(ethers.randomBytes(64));
      await orchestrator.connect(user1).executeChain(verifiers, [proof1], mockInputs);
      
      // User2 executes different chain
      const proof2 = ethers.hexlify(ethers.randomBytes(64));
      await orchestrator.connect(user2).executeChain(verifiers, [proof2], mockInputs);
      
      expect(await orchestrator.totalExecutions()).to.equal(2);
    });

    it("🔄 Should handle complex three-step chains", async function () {
      const verifiers = [
        await rwkvVerifier.getAddress(),
        await mambaVerifier.getAddress(),
        await xlstmVerifier.getAddress()
      ];
      
      const proofs = [
        ethers.hexlify(ethers.randomBytes(64)),
        ethers.hexlify(ethers.randomBytes(64)),
        ethers.hexlify(ethers.randomBytes(64))
      ];
      
      const tx = await orchestrator.executeChain(verifiers, proofs, mockInputs);
      const receipt = await tx.wait();
      
      // Should have 3 ChainStepCompleted events
      const stepEvents = receipt.logs.filter(
        log => log.fragment && log.fragment.name === "ChainStepCompleted"
      );
      expect(stepEvents.length).to.equal(3);
      
      // Final result should be successful
      const chainExecutedEvent = receipt.logs.find(
        log => log.fragment && log.fragment.name === "ChainExecuted"
      );
      const executionId = chainExecutedEvent.args[0];
      
      const [success, stepsCompleted, finalOutputs] = await orchestrator.getChainResult(executionId);
      expect(success).to.be.true;
      expect(stepsCompleted).to.equal(3);
      expect(finalOutputs.length).to.equal(10);
    });

    it("⛽ Should be gas efficient", async function () {
      const verifiers = [await rwkvVerifier.getAddress()];
      const proofs = [ethers.hexlify(ethers.randomBytes(64))];
      
      const tx = await orchestrator.executeChain(verifiers, proofs, mockInputs);
      const receipt = await tx.wait();
      
      console.log(`      ⛽ Gas used for single verification: ${receipt.gasUsed.toString()}`);
      
      // Should be reasonable gas usage (less than 500k for single step)
      expect(receipt.gasUsed).to.be.lessThan(500000);
    });
  });
});

describe("🎭 Gas Usage Analysis", function () {
  let contracts;
  
  beforeEach(async function () {
    const [owner] = await ethers.getSigners();
    
    // Deploy all contracts
    const RWKVVerifier = await ethers.getContractFactory("RWKVVerifier");
    const rwkvVerifier = await RWKVVerifier.deploy();
    
    const MambaVerifier = await ethers.getContractFactory("MambaVerifier");
    const mambaVerifier = await MambaVerifier.deploy();
    
    const xLSTMVerifier = await ethers.getContractFactory("xLSTMVerifier");
    const xlstmVerifier = await xLSTMVerifier.deploy();
    
    const ProofChainOrchestrator = await ethers.getContractFactory("ProofChainOrchestrator");
    const orchestrator = await ProofChainOrchestrator.deploy();
    
    contracts = { rwkvVerifier, mambaVerifier, xlstmVerifier, orchestrator };
  });

  it("📊 Should report deployment costs", async function () {
    console.log("      📋 Deployment Gas Costs:");
    console.log(`      🔗 RWKVVerifier: deployment gas`);
    console.log(`      🔗 MambaVerifier: deployment gas`);
    console.log(`      🔗 xLSTMVerifier: deployment gas`);
    console.log(`      🔗 Orchestrator: deployment gas`);
  });

  it("⛽ Should measure operation costs", async function () {
    const mockProof = ethers.hexlify(ethers.randomBytes(64));
    const mockInputs = [100, 200, 300, 400, 500];
    
    // Single verification
    const tx1 = await contracts.rwkvVerifier.verify(mockProof, mockInputs);
    const receipt1 = await tx1.wait();
    console.log(`      🔍 Single verification: ${receipt1.gasUsed} gas`);
    
    // Chain execution (3 steps)
    const verifiers = [
      await contracts.rwkvVerifier.getAddress(),
      await contracts.mambaVerifier.getAddress(),
      await contracts.xlstmVerifier.getAddress()
    ];
    const proofs = [
      ethers.hexlify(ethers.randomBytes(64)),
      ethers.hexlify(ethers.randomBytes(64)),
      ethers.hexlify(ethers.randomBytes(64))
    ];
    
    const tx2 = await contracts.orchestrator.executeChain(verifiers, proofs, mockInputs);
    const receipt2 = await tx2.wait();
    console.log(`      🔗 Chain execution (3 steps): ${receipt2.gasUsed} gas`);
    console.log(`      📈 Average per step: ${receipt2.gasUsed / 3n} gas`);
  });
});