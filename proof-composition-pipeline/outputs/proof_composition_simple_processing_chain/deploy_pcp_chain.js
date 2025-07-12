const hre = require("hardhat");
const fs = require("fs");

async function main() {
    console.log("🚀 Deploying Proof Composition Chain: simple_processing_chain");
    console.log("================================================================");
    
    // Deploy the chain verifier first
    console.log("📋 Deploying ChainVerifier...");
    const ChainVerifier = await hre.ethers.getContractFactory("ChainVerifier");
    const chainVerifier = await ChainVerifier.deploy();
    await chainVerifier.deployed();
    
    console.log("✅ ChainVerifier deployed to:", chainVerifier.address);
    console.log("📋 Transaction hash:", chainVerifier.deployTransaction.hash);
    console.log("⛽ Deploy gas used:", (await chainVerifier.deployTransaction.wait()).gasUsed.toString());
    
    // Deploy individual step verifiers
    const verifierAddresses = [];
    
    // Step 0: feature_extractor
    console.log("\n📊 Deploying Step 0 Verifier (feature_extractor)...");
    const Step0Verifier = await hre.ethers.getContractFactory("step00_feature_extractorVerifier");
    const step0Verifier = await Step0Verifier.deploy();
    await step0Verifier.deployed();
    verifierAddresses.push(step0Verifier.address);
    
    console.log("✅ Step 0 Verifier deployed to:", step0Verifier.address);
    console.log("📋 Transaction hash:", step0Verifier.deployTransaction.hash);
    console.log("⛽ Deploy gas used:", (await step0Verifier.deployTransaction.wait()).gasUsed.toString());
    
    // Step 1: classifier  
    console.log("\n📊 Deploying Step 1 Verifier (classifier)...");
    const Step1Verifier = await hre.ethers.getContractFactory("step01_classifierVerifier");
    const step1Verifier = await Step1Verifier.deploy();
    await step1Verifier.deployed();
    verifierAddresses.push(step1Verifier.address);
    
    console.log("✅ Step 1 Verifier deployed to:", step1Verifier.address);
    console.log("📋 Transaction hash:", step1Verifier.deployTransaction.hash);
    console.log("⛽ Deploy gas used:", (await step1Verifier.deployTransaction.wait()).gasUsed.toString());
    
    // Step 2: decision_maker
    console.log("\n📊 Deploying Step 2 Verifier (decision_maker)...");
    const Step2Verifier = await hre.ethers.getContractFactory("step02_decision_makerVerifier");
    const step2Verifier = await Step2Verifier.deploy();
    await step2Verifier.deployed();
    verifierAddresses.push(step2Verifier.address);
    
    console.log("✅ Step 2 Verifier deployed to:", step2Verifier.address);
    console.log("📋 Transaction hash:", step2Verifier.deployTransaction.hash);
    console.log("⛽ Deploy gas used:", (await step2Verifier.deployTransaction.wait()).gasUsed.toString());
    
    // Configure the chain verifier with individual verifier addresses
    console.log("\n🔗 Configuring ChainVerifier with individual verifiers...");
    const setAddressesTx = await chainVerifier.setVerifierAddresses(verifierAddresses);
    const setAddressesReceipt = await setAddressesTx.wait();
    
    console.log("✅ Verifier addresses configured");
    console.log("📋 Transaction hash:", setAddressesTx.hash);
    console.log("⛽ Config gas used:", setAddressesReceipt.gasUsed.toString());
    
    // Save deployment information
    const deploymentInfo = {
        chainId: "simple_processing_chain",
        chainVerifier: chainVerifier.address,
        individualVerifiers: verifierAddresses,
        contracts: {
            chainVerifier: {
                address: chainVerifier.address,
                deployTx: chainVerifier.deployTransaction.hash,
                gasUsed: (await chainVerifier.deployTransaction.wait()).gasUsed.toString()
            },
            step0Verifier: {
                address: step0Verifier.address,
                deployTx: step0Verifier.deployTransaction.hash,
                gasUsed: (await step0Verifier.deployTransaction.wait()).gasUsed.toString()
            },
            step1Verifier: {
                address: step1Verifier.address,
                deployTx: step1Verifier.deployTransaction.hash,
                gasUsed: (await step1Verifier.deployTransaction.wait()).gasUsed.toString()
            },
            step2Verifier: {
                address: step2Verifier.address,
                deployTx: step2Verifier.deployTransaction.hash,
                gasUsed: (await step2Verifier.deployTransaction.wait()).gasUsed.toString()
            }
        },
        timestamp: new Date().toISOString()
    };
    
    fs.writeFileSync("chain_deployment.json", JSON.stringify(deploymentInfo, null, 2));
    console.log("\n📁 Deployment info saved to chain_deployment.json");
    
    // Summary
    console.log("\n🎉 Chain Deployment Summary:");
    console.log("================================================================");
    console.log("🔗 Chain Verifier:", chainVerifier.address);
    console.log("📊 Step 0 (feature_extractor):", step0Verifier.address);
    console.log("📊 Step 1 (classifier):", step1Verifier.address);
    console.log("📊 Step 2 (decision_maker):", step2Verifier.address);
    console.log("📁 Deployment details saved in chain_deployment.json");
    console.log("🎯 Ready for chain verification testing!");
}

main()
    .then(() => process.exit(0))
    .catch((error) => {
        console.error("💥 Deployment failed:", error);
        process.exit(1);
    });