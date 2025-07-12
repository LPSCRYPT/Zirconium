const hre = require("hardhat");
const fs = require("fs");

async function main() {
    console.log("🚀 Deploying Proof Composition Chain: simple_processing_chain");
    console.log("================================================================");
    console.log("⏰ Start time:", new Date().toISOString());
    
    const [deployer] = await hre.ethers.getSigners();
    console.log("👤 Deploying with account:", deployer.address);
    
    const balance = await deployer.getBalance();
    console.log("💰 Account balance:", hre.ethers.formatEther(balance), "ETH");
    
    const network = await hre.ethers.provider.getNetwork();
    console.log("🌐 Network:", network.name, "| Chain ID:", network.chainId);
    console.log("================================================================");
    
    // Deploy the chain verifier first
    console.log("\n📋 Deploying ChainVerifier...");
    const ChainVerifier = await hre.ethers.getContractFactory("ChainVerifier");
    
    console.log("⏳ Sending deployment transaction...");
    const chainVerifier = await ChainVerifier.deploy();
    
    console.log("⏳ Waiting for deployment confirmation...");
    await chainVerifier.deployed();
    
    const chainVerifierReceipt = await chainVerifier.deployTransaction.wait();
    
    console.log("✅ ChainVerifier deployed successfully!");
    console.log("📍 Contract address:", chainVerifier.address);
    console.log("🔗 Transaction hash:", chainVerifier.deployTransaction.hash);
    console.log("⛽ Gas used:", chainVerifierReceipt.gasUsed.toString());
    console.log("💸 Gas price:", hre.ethers.utils.formatUnits(chainVerifier.deployTransaction.gasPrice, "gwei"), "gwei");
    console.log("🧾 Total cost:", hre.ethers.utils.formatEther(chainVerifierReceipt.gasUsed.mul(chainVerifier.deployTransaction.gasPrice)), "ETH");
    
    // Deploy individual step verifiers
    const verifierAddresses = [];
    const deploymentSummary = {
        chainVerifier: {
            address: chainVerifier.address,
            deployTx: chainVerifier.deployTransaction.hash,
            gasUsed: chainVerifierReceipt.gasUsed.toString(),
            gasPrice: chainVerifier.deployTransaction.gasPrice.toString()
        },
        stepVerifiers: []
    };
    
    const steps = [
        { name: "feature_extractor", contractName: "step00_feature_extractorVerifier" },
        { name: "classifier", contractName: "step01_classifierVerifier" },
        { name: "decision_maker", contractName: "step02_decision_makerVerifier" }
    ];
    
    for (let i = 0; i < steps.length; i++) {
        const step = steps[i];
        console.log(`\n📊 Deploying Step ${i} Verifier (${step.name})...`);
        
        const StepVerifier = await hre.ethers.getContractFactory(step.contractName);
        console.log("⏳ Sending deployment transaction...");
        const stepVerifier = await StepVerifier.deploy();
        
        console.log("⏳ Waiting for deployment confirmation...");
        await stepVerifier.deployed();
        
        const stepVerifierReceipt = await stepVerifier.deployTransaction.wait();
        verifierAddresses.push(stepVerifier.address);
        
        console.log(`✅ Step ${i} Verifier deployed successfully!`);
        console.log("📍 Contract address:", stepVerifier.address);
        console.log("🔗 Transaction hash:", stepVerifier.deployTransaction.hash);
        console.log("⛽ Gas used:", stepVerifierReceipt.gasUsed.toString());
        console.log("💸 Gas price:", hre.ethers.utils.formatUnits(stepVerifier.deployTransaction.gasPrice, "gwei"), "gwei");
        console.log("🧾 Cost:", hre.ethers.utils.formatEther(stepVerifierReceipt.gasUsed.mul(stepVerifier.deployTransaction.gasPrice)), "ETH");
        
        deploymentSummary.stepVerifiers.push({
            step: i,
            name: step.name,
            address: stepVerifier.address,
            deployTx: stepVerifier.deployTransaction.hash,
            gasUsed: stepVerifierReceipt.gasUsed.toString(),
            gasPrice: stepVerifier.deployTransaction.gasPrice.toString()
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
    console.log("💸 Gas price:", hre.ethers.utils.formatUnits(setAddressesTx.gasPrice, "gwei"), "gwei");
    console.log("🧾 Cost:", hre.ethers.utils.formatEther(setAddressesReceipt.gasUsed.mul(setAddressesTx.gasPrice)), "ETH");
    
    // Calculate total deployment costs
    const totalGasUsed = deploymentSummary.stepVerifiers.reduce((total, step) => {
        return total.add(hre.ethers.BigNumber.from(step.gasUsed));
    }, chainVerifierReceipt.gasUsed.add(setAddressesReceipt.gasUsed));
    
    const avgGasPrice = hre.ethers.BigNumber.from(chainVerifier.deployTransaction.gasPrice);
    const totalCost = totalGasUsed.mul(avgGasPrice);
    
    // Save comprehensive deployment information
    const deploymentInfo = {
        chainId: "simple_processing_chain",
        network: {
            name: network.name,
            chainId: network.chainId
        },
        deployer: {
            address: deployer.address,
            balanceAfter: hre.ethers.utils.formatEther(await deployer.getBalance())
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
            totalCostETH: hre.ethers.utils.formatEther(totalCost),
            avgGasPriceGwei: hre.ethers.utils.formatUnits(avgGasPrice, "gwei")
        },
        timestamp: new Date().toISOString()
    };
    
    fs.writeFileSync("chain_deployment_detailed.json", JSON.stringify(deploymentInfo, null, 2));
    console.log("\n📁 Detailed deployment info saved to chain_deployment_detailed.json");
    
    // Also save simple format for compatibility
    const simpleDeploymentInfo = {
        chainId: "simple_processing_chain",
        chainVerifier: chainVerifier.address,
        individualVerifiers: verifierAddresses,
        timestamp: new Date().toISOString()
    };
    
    fs.writeFileSync("chain_deployment.json", JSON.stringify(simpleDeploymentInfo, null, 2));
    console.log("📁 Simple deployment info saved to chain_deployment.json");
    
    // Final summary
    console.log("\n🎉 DEPLOYMENT SUMMARY");
    console.log("================================================================");
    console.log("🔗 Chain Verifier:", chainVerifier.address);
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