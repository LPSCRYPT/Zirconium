const hre = require("hardhat");
const fs = require("fs");

async function main() {
    console.log("🚀 Deploying Proof Composition Chain: simple_processing_chain");
    console.log("================================================================");
    console.log("⏰ Start time:", new Date().toISOString());
    
    const [deployer] = await hre.ethers.getSigners();
    console.log("👤 Deploying with account:", deployer.address);
    
    const balance = await deployer.provider.getBalance(deployer.address);
    console.log("💰 Account balance:", hre.ethers.formatEther(balance), "ETH");
    
    const network = await deployer.provider.getNetwork();
    console.log("🌐 Network:", network.name, "| Chain ID:", network.chainId);
    console.log("================================================================");
    
    // Deploy the chain verifier first
    console.log("\n📋 Deploying ChainVerifier...");
    const ChainVerifier = await hre.ethers.getContractFactory("ChainVerifier");
    
    console.log("⏳ Sending deployment transaction...");
    const chainVerifier = await ChainVerifier.deploy();
    
    console.log("⏳ Waiting for deployment confirmation...");
    await chainVerifier.waitForDeployment();
    
    const chainVerifierAddress = await chainVerifier.getAddress();
    const deployTx = chainVerifier.deploymentTransaction();
    const receipt = await deployTx.wait();
    
    console.log("✅ ChainVerifier deployed successfully!");
    console.log("📍 Contract address:", chainVerifierAddress);
    console.log("🔗 Transaction hash:", deployTx.hash);
    console.log("⛽ Gas used:", receipt.gasUsed.toString());
    console.log("💸 Gas price:", hre.ethers.formatUnits(deployTx.gasPrice, "gwei"), "gwei");
    console.log("🧾 Total cost:", hre.ethers.formatEther(receipt.gasUsed * deployTx.gasPrice), "ETH");
    
    // Deploy individual step verifiers
    const verifierAddresses = [];
    const deploymentSummary = {
        chainVerifier: {
            address: chainVerifierAddress,
            deployTx: deployTx.hash,
            gasUsed: receipt.gasUsed.toString(),
            gasPrice: deployTx.gasPrice.toString()
        },
        stepVerifiers: []
    };
    
    const steps = [
        { name: "feature_extractor", contractName: "contracts/step00_feature_extractorVerifier.sol:Halo2Verifier" },
        { name: "classifier", contractName: "contracts/step01_classifierVerifier.sol:Halo2Verifier" },
        { name: "decision_maker", contractName: "contracts/step02_decision_makerVerifier.sol:Halo2Verifier" }
    ];
    
    for (let i = 0; i < steps.length; i++) {
        const step = steps[i];
        console.log(`\n📊 Deploying Step ${i} Verifier (${step.name})...`);
        
        const StepVerifier = await hre.ethers.getContractFactory(step.contractName);
        console.log("⏳ Sending deployment transaction...");
        const stepVerifier = await StepVerifier.deploy();
        
        console.log("⏳ Waiting for deployment confirmation...");
        await stepVerifier.waitForDeployment();
        
        const stepVerifierAddress = await stepVerifier.getAddress();
        const stepDeployTx = stepVerifier.deploymentTransaction();
        const stepReceipt = await stepDeployTx.wait();
        
        verifierAddresses.push(stepVerifierAddress);
        
        console.log(`✅ Step ${i} Verifier deployed successfully!`);
        console.log("📍 Contract address:", stepVerifierAddress);
        console.log("🔗 Transaction hash:", stepDeployTx.hash);
        console.log("⛽ Gas used:", stepReceipt.gasUsed.toString());
        console.log("💸 Gas price:", hre.ethers.formatUnits(stepDeployTx.gasPrice, "gwei"), "gwei");
        console.log("🧾 Cost:", hre.ethers.formatEther(stepReceipt.gasUsed * stepDeployTx.gasPrice), "ETH");
        
        deploymentSummary.stepVerifiers.push({
            step: i,
            name: step.name,
            address: stepVerifierAddress,
            deployTx: stepDeployTx.hash,
            gasUsed: stepReceipt.gasUsed.toString(),
            gasPrice: stepDeployTx.gasPrice.toString()
        });
    }
    
    // Configure the chain verifier with individual verifier addresses
    console.log("\n🔗 Configuring ChainVerifier with individual verifiers...");
    console.log("⏳ Sending configuration transaction...");
    const setAddressesTx = await chainVerifier.setVerifierAddresses(verifierAddresses);
    
    console.log("⏳ Waiting for configuration confirmation...");
    const setAddressesReceipt = await setAddressesTx.wait();
    
    console.log("✅ Verifier addresses configured successfully!");
    console.log("🔗 Transaction hash:", setAddressesTx.hash);
    console.log("⛽ Gas used:", setAddressesReceipt.gasUsed.toString());
    console.log("💸 Gas price:", hre.ethers.formatUnits(setAddressesTx.gasPrice, "gwei"), "gwei");
    console.log("🧾 Cost:", hre.ethers.formatEther(setAddressesReceipt.gasUsed * setAddressesTx.gasPrice), "ETH");
    
    // Calculate total deployment costs
    let totalGasUsed = receipt.gasUsed + setAddressesReceipt.gasUsed;
    for (const step of deploymentSummary.stepVerifiers) {
        totalGasUsed += BigInt(step.gasUsed);
    }
    
    const avgGasPrice = deployTx.gasPrice;
    const totalCost = totalGasUsed * avgGasPrice;
    
    // Save comprehensive deployment information
    const deploymentInfo = {
        chainId: "simple_processing_chain",
        network: {
            name: network.name,
            chainId: network.chainId.toString()
        },
        deployer: {
            address: deployer.address,
            balanceAfter: hre.ethers.formatEther(await deployer.provider.getBalance(deployer.address))
        },
        chainVerifier: deploymentSummary.chainVerifier,
        stepVerifiers: deploymentSummary.stepVerifiers,
        configuration: {
            setAddressesTx: setAddressesTx.hash,
            gasUsed: setAddressesReceipt.gasUsed.toString(),
            gasPrice: setAddressesTx.gasPrice.toString()
        },
        summary: {
            totalContracts: 1 + steps.length,
            totalTransactions: 1 + steps.length + 1, // deployments + configuration
            totalGasUsed: totalGasUsed.toString(),
            totalCostETH: hre.ethers.formatEther(totalCost),
            avgGasPriceGwei: hre.ethers.formatUnits(avgGasPrice, "gwei")
        },
        timestamp: new Date().toISOString()
    };
    
    fs.writeFileSync("chain_deployment_detailed.json", JSON.stringify(deploymentInfo, null, 2));
    console.log("\n📁 Detailed deployment info saved to chain_deployment_detailed.json");
    
    // Also save simple format for compatibility
    const simpleDeploymentInfo = {
        chainId: "simple_processing_chain",
        chainVerifier: chainVerifierAddress,
        individualVerifiers: verifierAddresses,
        timestamp: new Date().toISOString()
    };
    
    fs.writeFileSync("chain_deployment.json", JSON.stringify(simpleDeploymentInfo, null, 2));
    console.log("📁 Simple deployment info saved to chain_deployment.json");
    
    // Final summary
    console.log("\n🎉 DEPLOYMENT SUMMARY");
    console.log("================================================================");
    console.log("🔗 Chain Verifier:", chainVerifierAddress);
    console.log("📊 Step Verifiers:");
    for (let i = 0; i < verifierAddresses.length; i++) {
        console.log(`   ${i}: ${steps[i].name} -> ${verifierAddresses[i]}`);
    }
    console.log("📊 Total Contracts Deployed:", deploymentInfo.summary.totalContracts);
    console.log("🔗 Total Transactions:", deploymentInfo.summary.totalTransactions);
    console.log("⛽ Total Gas Used:", deploymentInfo.summary.totalGasUsed);
    console.log("💰 Total Cost:", deploymentInfo.summary.totalCostETH, "ETH");
    console.log("💸 Average Gas Price:", deploymentInfo.summary.avgGasPriceGwei, "gwei");
    console.log("⏰ Completion time:", new Date().toISOString());
    console.log("🎯 Ready for chain verification testing!");
    console.log("================================================================");
}

main()
    .then(() => process.exit(0))
    .catch((error) => {
        console.error("💥 Deployment failed:", error);
        console.error("🔍 Stack trace:", error.stack);
        process.exit(1);
    });