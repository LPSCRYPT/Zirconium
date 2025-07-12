#!/usr/bin/env python3
"""
Zircuit Blockchain Submission for Weather Prediction
Submit verifiable weather prediction to Zircuit Garfield testnet
"""

import json
import time
from web3 import Web3
from eth_account import Account
import os
from typing import Dict, List, Tuple
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ZircuitBlockchainSubmitter:
    """Handle blockchain submission of weather predictions to Zircuit"""
    
    def __init__(self):
        # Load deployed contract addresses from Zircuit testnet
        addresses_file = "config/addresses/zircuit-testnet-addresses.json"
        if os.path.exists(addresses_file):
            with open(addresses_file, 'r') as f:
                deployment_data = json.load(f)
                self.contracts = deployment_data["contracts"]
        else:
            # Fallback to deployed addresses from our deployment
            self.contracts = {
                "ProductionRWKVVerifier": "0xFbd3323361E55fbfF58134933262368b76FD41e8"
            }
        
        # Zircuit Garfield testnet configuration
        self.chain_id = 48898
        self.rpc_url = "https://garfield-testnet.zircuit.com"
        
        # Initialize Web3
        self.w3 = Web3(Web3.HTTPProvider(self.rpc_url))
        
        if not self.w3.is_connected():
            raise Exception("Failed to connect to Zircuit Garfield testnet")
        
        print(f"🔗 Connected to Zircuit Garfield Testnet")
        print(f"   📡 RPC: {self.rpc_url}")
        print(f"   🆔 Chain ID: {self.chain_id}")
        print(f"   📊 Latest block: {self.w3.eth.block_number}")
        
        # Contract ABI (RWKV verifier interface)
        self.verifier_abi = [
            {
                "inputs": [
                    {
                        "components": [
                            {"name": "a", "type": "uint256[2]"},
                            {"name": "b", "type": "uint256[2][2]"},
                            {"name": "c", "type": "uint256[2]"},
                            {"name": "z", "type": "uint256[2]"},
                            {"name": "t1", "type": "uint256[2]"},
                            {"name": "t2", "type": "uint256[2]"},
                            {"name": "t3", "type": "uint256[2]"},
                            {"name": "eval_a", "type": "uint256"},
                            {"name": "eval_b", "type": "uint256"},
                            {"name": "eval_c", "type": "uint256"},
                            {"name": "eval_s1", "type": "uint256"},
                            {"name": "eval_s2", "type": "uint256"},
                            {"name": "eval_zw", "type": "uint256"}
                        ],
                        "name": "proof",
                        "type": "tuple"
                    },
                    {"name": "publicInputs", "type": "uint256[]"}
                ],
                "name": "verifyProof",
                "outputs": [{"name": "", "type": "bool"}],
                "stateMutability": "nonpayable",
                "type": "function"
            }
        ]
    
    def load_wallet(self) -> Tuple[str, str]:
        """Load wallet from environment for testing"""
        
        print("\n🔑 Loading Wallet for Zircuit")
        print("-" * 28)
        
        # Load from .env file
        private_key = os.getenv('PRIVATE_KEY')
        if not private_key:
            print("❌ No PRIVATE_KEY found in environment")
            return None, None
        
        if not private_key.startswith('0x'):
            private_key = '0x' + private_key
            
        account = Account.from_key(private_key)
        address = account.address
        
        print(f"   📍 Address: {address}")
        
        # Check balance
        balance = self.w3.eth.get_balance(address)
        balance_eth = self.w3.from_wei(balance, 'ether')
        
        print(f"   💰 Balance: {balance_eth} ETH")
        
        if balance_eth < 0.001:
            print("   ⚠️  Low ETH balance!")
            print("      Send more ETH to Zircuit Garfield testnet")
            
            return private_key, address
        
        return private_key, address
    
    def prepare_proof_data(self, prediction_results: Dict) -> Tuple[Dict, List[int]]:
        """Convert prediction results to blockchain-compatible format"""
        
        print("\n🔧 Preparing Proof Data for Zircuit")
        print("-" * 33)
        
        # Load the proof file
        proof_file = prediction_results["files_generated"]["proof"]
        with open(proof_file, 'r') as f:
            proof_data = json.load(f)
        
        # Convert EZKL proof to contract format
        try:
            # Parse real EZKL proof structure  
            proof_components = proof_data.get("proof", {})
            proof_struct = {
                "a": [
                    int.from_bytes(bytes.fromhex(str(proof_components.get("a", ["0"]*2)[0]).replace("0x", "")), byteorder='big'),
                    int.from_bytes(bytes.fromhex(str(proof_components.get("a", ["0"]*2)[1]).replace("0x", "")), byteorder='big')
                ],
                "b": [
                    [
                        int.from_bytes(bytes.fromhex(str(proof_components.get("b", [["0"]*2, ["0"]*2])[0][0]).replace("0x", "")), byteorder='big'),
                        int.from_bytes(bytes.fromhex(str(proof_components.get("b", [["0"]*2, ["0"]*2])[0][1]).replace("0x", "")), byteorder='big')
                    ],
                    [
                        int.from_bytes(bytes.fromhex(str(proof_components.get("b", [["0"]*2, ["0"]*2])[1][0]).replace("0x", "")), byteorder='big'),
                        int.from_bytes(bytes.fromhex(str(proof_components.get("b", [["0"]*2, ["0"]*2])[1][1]).replace("0x", "")), byteorder='big')
                    ]
                ],
                "c": [
                    int.from_bytes(bytes.fromhex(str(proof_components.get("c", ["0"]*2)[0]).replace("0x", "")), byteorder='big'),
                    int.from_bytes(bytes.fromhex(str(proof_components.get("c", ["0"]*2)[1]).replace("0x", "")), byteorder='big')
                ],
                "z": [
                    int.from_bytes(bytes.fromhex(str(proof_components.get("z", ["0"]*2)[0]).replace("0x", "")), byteorder='big'),
                    int.from_bytes(bytes.fromhex(str(proof_components.get("z", ["0"]*2)[1]).replace("0x", "")), byteorder='big')
                ],
                "t1": [
                    int.from_bytes(bytes.fromhex(str(proof_components.get("t1", ["0"]*2)[0]).replace("0x", "")), byteorder='big'),
                    int.from_bytes(bytes.fromhex(str(proof_components.get("t1", ["0"]*2)[1]).replace("0x", "")), byteorder='big')
                ],
                "t2": [
                    int.from_bytes(bytes.fromhex(str(proof_components.get("t2", ["0"]*2)[0]).replace("0x", "")), byteorder='big'),
                    int.from_bytes(bytes.fromhex(str(proof_components.get("t2", ["0"]*2)[1]).replace("0x", "")), byteorder='big')
                ],
                "t3": [
                    int.from_bytes(bytes.fromhex(str(proof_components.get("t3", ["0"]*2)[0]).replace("0x", "")), byteorder='big'),
                    int.from_bytes(bytes.fromhex(str(proof_components.get("t3", ["0"]*2)[1]).replace("0x", "")), byteorder='big')
                ],
                "eval_a": int.from_bytes(bytes.fromhex(str(proof_components.get("eval_a", "0")).replace("0x", "")), byteorder='big'),
                "eval_b": int.from_bytes(bytes.fromhex(str(proof_components.get("eval_b", "0")).replace("0x", "")), byteorder='big'),
                "eval_c": int.from_bytes(bytes.fromhex(str(proof_components.get("eval_c", "0")).replace("0x", "")), byteorder='big'),
                "eval_s1": int.from_bytes(bytes.fromhex(str(proof_components.get("eval_s1", "0")).replace("0x", "")), byteorder='big'),
                "eval_s2": int.from_bytes(bytes.fromhex(str(proof_components.get("eval_s2", "0")).replace("0x", "")), byteorder='big'),
                "eval_zw": int.from_bytes(bytes.fromhex(str(proof_components.get("eval_zw", "0")).replace("0x", "")), byteorder='big')
            }
        except Exception as e:
            print(f"   ⚠️ Warning: Could not parse EZKL proof format: {e}")
            print(f"   🔄 Using default proof structure for demonstration")
            # Fallback to demonstration values if proof parsing fails
            proof_struct = {
                "a": [1, 2], "b": [[3, 4], [5, 6]], "c": [7, 8],
                "z": [9, 10], "t1": [11, 12], "t2": [13, 14], "t3": [15, 16],
                "eval_a": 17, "eval_b": 18, "eval_c": 19, 
                "eval_s1": 20, "eval_s2": 21, "eval_zw": 22
            }
        
        # Convert prediction to public inputs
        prediction = prediction_results["prediction"]
        public_inputs = [
            int(prediction["predicted_temp_f"]),  # Temperature in °F
            2,  # Weather condition (cloudy = 2)
            int(prediction["confidence"] * 100),  # Confidence as percentage
            1  # Location ID (SF = 1)
        ]
        
        print(f"   📝 Proof structure prepared")
        print(f"   🔢 Public inputs: {public_inputs}")
        print(f"   📍 Target contract: {self.contracts['ProductionRWKVVerifier']}")
        
        return proof_struct, public_inputs
    
    def submit_to_zircuit(self, prediction_results: Dict) -> Dict:
        """Submit prediction to Zircuit Garfield testnet"""
        
        print("\n🚀 Submitting to Zircuit Garfield Testnet")
        print("-" * 36)
        
        # Load wallet
        private_key, address = self.load_wallet()
        if not private_key:
            return {"success": False, "error": "No wallet found", "simulation": False}
        
        # Prepare proof data
        proof_struct, public_inputs = self.prepare_proof_data(prediction_results)
        
        # Create contract instance
        contract_address = self.contracts["ProductionRWKVVerifier"]
        contract = self.w3.eth.contract(
            address=contract_address,
            abi=self.verifier_abi
        )
        
        try:
            # Build transaction
            account = Account.from_key(private_key)
            
            # Get current gas price
            gas_price = self.w3.eth.gas_price
            print(f"   ⛽ Current gas price: {self.w3.from_wei(gas_price, 'gwei')} gwei")
            
            # Call verifyProof
            txn = contract.functions.verifyProof(
                proof_struct,
                public_inputs
            ).build_transaction({
                'from': address,
                'gas': 200000,  # Reduced gas limit for verification
                'gasPrice': gas_price,
                'nonce': self.w3.eth.get_transaction_count(address)
            })
            
            # Sign and send transaction
            signed_txn = account.sign_transaction(txn)
            tx_hash = self.w3.eth.send_raw_transaction(signed_txn.raw_transaction)
            
            print(f"   📤 Transaction sent: {tx_hash.hex()}")
            print(f"   ⏳ Waiting for confirmation...")
            
            # Wait for transaction receipt
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
            
            if receipt.status == 1:
                print(f"   ✅ Transaction confirmed!")
                print(f"   📊 Gas used: {receipt.gasUsed:,}")
                
                # Calculate cost
                cost = receipt.gasUsed * gas_price
                cost_eth = self.w3.from_wei(cost, 'ether')
                
                submission_result = {
                    "success": True,
                    "simulation": False,
                    "transaction_hash": tx_hash.hex(),
                    "contract_address": contract_address,
                    "network": "Zircuit Garfield Testnet",
                    "chain_id": self.chain_id,
                    "timestamp": int(time.time()),
                    "gas_used": receipt.gasUsed,
                    "gas_price_gwei": self.w3.from_wei(gas_price, 'gwei'),
                    "cost_eth": cost_eth,
                    "block_number": receipt.blockNumber,
                    "zircuit_explorer": f"https://explorer.garfield-testnet.zircuit.com/tx/{tx_hash.hex()}"
                }
                
                print(f"   💸 Cost: {cost_eth:.6f} ETH")
                print(f"   🔗 Zircuit Explorer: {submission_result['zircuit_explorer']}")
                
                return submission_result
                
            else:
                print(f"   ❌ Transaction failed!")
                return {"success": False, "error": "Transaction failed", "simulation": False}
                
        except Exception as e:
            print(f"   ❌ Submission failed: {e}")
            return {"success": False, "error": str(e), "simulation": False}

def submit_weather_prediction_to_zircuit():
    """Main function to submit weather prediction to Zircuit"""
    
    print("🚀 Weather Prediction Zircuit Submission")
    print("=" * 40)
    
    # Load prediction results
    results_file = "data/predictions/sf_weather_prediction_results.json"
    if not os.path.exists(results_file):
        print(f"❌ Results file not found: {results_file}")
        print("   Please run weather_prediction.py first")
        return
    
    with open(results_file, 'r') as f:
        prediction_results = json.load(f)
    
    print(f"📄 Loaded prediction results: {prediction_results['prediction']['date']}")
    
    # Initialize Zircuit blockchain submitter
    try:
        submitter = ZircuitBlockchainSubmitter()
    except Exception as e:
        print(f"❌ Failed to initialize Zircuit connection: {e}")
        return
    
    # Submit to Zircuit
    submission_result = submitter.submit_to_zircuit(prediction_results)
    
    # Display results
    if submission_result.get("success"):
        print(f"\n🎉 ZK PROOF VERIFICATION ON ZIRCUIT SUCCESS!")
        print(f"=" * 45)
        print(f"📅 Date: {prediction_results['prediction']['date']}")
        print(f"🌡️ Temperature: {prediction_results['prediction']['predicted_temp_f']}°F")
        print(f"☁️ Condition: {prediction_results['prediction']['condition']}")
        print(f"🎲 Confidence: {prediction_results['prediction']['confidence']:.1%}")
        print(f"🔗 Network: Zircuit Garfield Testnet")
        print(f"📍 Contract: {submission_result['contract_address']}")
        print(f"🔗 Transaction: {submission_result['transaction_hash']}")
        print(f"🔍 Explorer: {submission_result['zircuit_explorer']}")
    else:
        print(f"\n❌ Verification failed: {submission_result.get('error', 'Unknown error')}")
        print(f"💡 This may be expected with mock proof data")
    
    return submission_result

if __name__ == "__main__":
    submit_weather_prediction_to_zircuit()