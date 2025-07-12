const { ethers } = require("hardhat");
const fs = require('fs');

async function main() {
  console.log("🔍 Debugging EZKL Proof Verification");
  
  // Get latest deployment
  const deploymentFiles = fs.readdirSync('config/deployments')
    .filter(f => f.startsWith('localhost-minimal-'))
    .sort()
    .reverse();
  
  if (deploymentFiles.length === 0) {
    console.log("❌ No deployments found");
    return;
  }
  
  const deploymentData = JSON.parse(fs.readFileSync(`config/deployments/${deploymentFiles[0]}`, 'utf8'));
  console.log("📄 Using deployment:", deploymentFiles[0]);
  
  // Connect to RWKV verifier wrapper
  const rwkvAddress = deploymentData.contracts.RWKVVerifier;
  const RWKVVerifier = await ethers.getContractFactory("RWKVVerifierWrapper");
  const rwkvVerifier = RWKVVerifier.attach(rwkvAddress);
  
  console.log("🔗 Connected to RWKVVerifier at:", rwkvAddress);
  console.log("📊 Architecture:", await rwkvVerifier.getArchitecture());
  console.log("🔗 Underlying EZKL verifier:", await rwkvVerifier.getEZKLVerifier());
  
  // Load proof data
  const proofData = JSON.parse(fs.readFileSync('ezkl_workspace/rwkv_simple/proof.json', 'utf8'));
  
  // Convert proof to bytes using hex_proof
  const hexProof = proofData.hex_proof.startsWith('0x') ? proofData.hex_proof.slice(2) : proofData.hex_proof;
  const proofBytes = '0x' + hexProof;
  
  // Extract public inputs (convert hex to decimal)
  const publicInputs = proofData.instances[0].map(hex => BigInt('0x' + hex));
  
  console.log("📄 Proof size:", proofBytes.length, "bytes");
  console.log("📊 Public inputs:", publicInputs.length, "values");
  console.log("📊 First 5 inputs:", publicInputs.slice(0, 5).map(x => x.toString()));
  
  // Try to call the underlying EZKL verifier directly
  const ezkl_address = await rwkvVerifier.getEZKLVerifier();
  const EZKLVerifier = await ethers.getContractFactory("RWKVVerifier");
  const ezklVerifier = EZKLVerifier.attach(ezkl_address);
  
  console.log("\n🧪 Testing direct EZKL verifier call...");
  
  try {
    // Test gas estimation first
    const gasEstimate = await ezklVerifier.verifyProof.estimateGas(proofBytes, publicInputs);
    console.log("⛽ Gas estimate:", gasEstimate.toString());
    
    // Try the actual call
    const result = await ezklVerifier.verifyProof(proofBytes, publicInputs);
    console.log("✅ Direct EZKL verification result:", result);
    
  } catch (error) {
    console.log("❌ Direct EZKL verification failed:", error.message);
    
    if (error.message.includes('revert')) {
      console.log("🔍 This suggests the proof/inputs are invalid for the verifier");
    }
  }
  
  console.log("\n🧪 Testing wrapper verifier call...");
  
  try {
    // Test gas estimation first
    const gasEstimate = await rwkvVerifier.verify.estimateGas(proofBytes, publicInputs);
    console.log("⛽ Gas estimate:", gasEstimate.toString());
    
    // Try the actual call
    const result = await rwkvVerifier.verify(proofBytes, publicInputs);
    console.log("✅ Wrapper verification result:", result);
    
  } catch (error) {
    console.log("❌ Wrapper verification failed:", error.message);
  }
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error("❌ Debug failed:", error);
    process.exit(1);
  });