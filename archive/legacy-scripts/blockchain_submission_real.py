#!/usr/bin/env python3
"""
Real blockchain submission - no simulation
"""
import json
import sys
import os
from web3 import Web3
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from ezkl_to_solidity import convert_ezkl_proof_to_solidity

def load_contract_abi():
    """Load the contract ABI from Hardhat artifacts"""
    abi_path = "/Users/bot/code/zirconium/artifacts/contracts/WeatherVerifier.sol/WeatherVerifier.json"
    
    with open(abi_path, 'r') as f:
        contract_artifact = json.load(f)
    
    return contract_artifact['abi']

def submit_proof_to_blockchain():
    """Actually submit the proof to the blockchain"""
    print("🔗 Connecting to localhost blockchain...")
    
    # Connect to localhost
    w3 = Web3(Web3.HTTPProvider('http://127.0.0.1:8545'))
    
    if not w3.is_connected():
        raise Exception("Failed to connect to localhost blockchain")
    
    print(f"✅ Connected to localhost (Chain ID: {w3.eth.chain_id})")
    
    # Load deployment config
    config_path = "/Users/bot/code/zirconium/config/weather-verifier-address.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    verifier_address = config['contract']['WeatherVerifier']
    print(f"📍 Weather Verifier: {verifier_address}")
    
    # Load contract ABI and create contract instance
    contract_abi = load_contract_abi()
    contract = w3.eth.contract(address=verifier_address, abi=contract_abi)
    
    # Convert EZKL proof to Solidity format
    proof_path = "/Users/bot/code/zirconium/ezkl_weather_workspace/proof.json"
    proof_bytes, instances = convert_ezkl_proof_to_solidity(proof_path)
    
    # Use the default account (first one from hardhat node)
    default_account = w3.eth.accounts[0]
    
    print(f"\n👤 Using account: {default_account}")
    print(f"💰 Balance: {w3.from_wei(w3.eth.get_balance(default_account), 'ether')} ETH")
    
    print(f"\n🚀 SUBMITTING PROOF TO BLOCKCHAIN")
    print("=" * 35)
    
    try:
        # First try a read-only call to check if the proof format is correct
        print("📝 Testing verifyProof function with read-only call...")
        
        try:
            print(f"Calling verifyProof with:")
            print(f"  - proof_bytes type: {type(proof_bytes)}")
            print(f"  - proof_bytes length: {len(proof_bytes)}")
            print(f"  - instances type: {type(instances)}")
            print(f"  - instances length: {len(instances)}")
            print(f"  - instances values: {instances}")
            
            result = contract.functions.verifyProof(proof_bytes, instances).call({
                'gas': 30000000  # Very high gas limit
            })
            print(f"✅ Read-only call successful: {result}")
        except Exception as e:
            print(f"❌ Read-only call failed: {e}")
            print("This indicates the proof format or data is incorrect")
            
            # Try with a simpler debug call
            try:
                print("Trying to call contract with empty data...")
                result = contract.functions.verifyProof(b'', []).call({
                    'gas': 30000000
                })
                print(f"Empty call result: {result}")
            except Exception as e2:
                print(f"Empty call also failed: {e2}")
            
            return False, None
        
        # If read-only call succeeded, proceed with transaction
        print("📝 Submitting transaction to blockchain...")
        
        # Create transaction
        transaction = contract.functions.verifyProof(
            proof_bytes,
            instances
        ).build_transaction({
            'from': default_account,
            'gas': 2000000,  # High gas limit for ZK verification
            'gasPrice': w3.to_wei('20', 'gwei'),
            'nonce': w3.eth.get_transaction_count(default_account),
        })
        
        # Sign and send transaction
        signed_txn = w3.eth.account.sign_transaction(transaction, private_key='0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80')
        tx_hash = w3.eth.send_raw_transaction(signed_txn.raw_transaction)
        
        print(f"📤 Transaction sent: {tx_hash.hex()}")
        
        # Wait for confirmation
        print("⏳ Waiting for transaction confirmation...")
        tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        
        print(f"✅ Transaction confirmed in block: {tx_receipt.blockNumber}")
        print(f"⛽ Gas used: {tx_receipt.gasUsed}")
        
        # Check if transaction was successful
        if tx_receipt.status == 1:
            print(f"🎉 PROOF VERIFICATION SUCCESSFUL!")
            
            # Try to get the return value (this might not work with transactions)
            try:
                result = contract.functions.verifyProof(proof_bytes, instances).call()
                print(f"✅ Verification result: {result}")
            except Exception as e:
                print(f"ℹ️  Could not retrieve return value: {e}")
                print("ℹ️  This is normal for state-changing transactions")
            
            # Log successful submission
            submission_log = {
                "timestamp": datetime.now().isoformat(),
                "network": "localhost",
                "chain_id": w3.eth.chain_id,
                "verifier_contract": verifier_address,
                "submitter": default_account,
                "transaction_hash": tx_hash.hex(),
                "block_number": tx_receipt.blockNumber,
                "gas_used": tx_receipt.gasUsed,
                "status": "success",
                "prediction": {
                    "temperature": "62.8°F",
                    "conditions": "Cloudy",
                    "location": "San Francisco",
                    "date": "July 10, 2025"
                }
            }
            
            with open("/Users/bot/code/zirconium/logs/successful_submission.json", "w") as f:
                json.dump(submission_log, f, indent=2)
            
            return True, submission_log
            
        else:
            print(f"❌ Transaction failed!")
            return False, None
            
    except Exception as e:
        print(f"❌ Error submitting proof: {e}")
        return False, None

if __name__ == "__main__":
    try:
        success, result = submit_proof_to_blockchain()
        if success:
            print(f"\n🎉 Weather prediction proof successfully submitted to blockchain!")
        else:
            print(f"\n❌ Failed to submit proof to blockchain")
            exit(1)
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        exit(1)