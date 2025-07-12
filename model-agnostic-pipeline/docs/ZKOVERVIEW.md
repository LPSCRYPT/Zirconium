Zero-Knowledge Proof Types, Examples, and Composite zkML Systems

0. Notation and Foundations

0.1 Basic Logical Structure

A zero-knowledge proof (ZKP) allows a prover to convince a verifier that a statement is true without revealing why it's true.

Basic format:

\exists x : f(x) = y

This means: There exists an input x such that when applying function f, the result is y.

0.2 Common Symbols

Symbol

Meaning



"There exists"



"For all"



Secret or witness value(s)



Computable function or program output



Commitment to a value



A known or committed set



Membership ("is in")



Logical AND



Logical OR



Logical implication

1. Core Zero-Knowledge Statement Types

1.1 Proof of Knowledge

Statement: I know a secret w such that a relation R(x, w) holds.

LaTeX:

\text{Given } R : \mathcal{X} \times \mathcal{W} \rightarrow \{0,1\}, \text{ prove: } \exists w \in \mathcal{W} \text{ s.t. } R(x, w) = 1

Example: Prove you know the discrete log of a public key: 

1.2 Set Membership

Statement: x is a member of a set S.

LaTeX:

\text{Prove: } x \in S

Example: Prove your identity hash is in a Merkle tree representing an allowlist.

1.3 Range Proofs

Statement: A secret number is within a range.

LaTeX:

\text{Prove: } l \leq a \leq h

Example: Prove your age is in the range [18, 120].

1.4 Committed Value Statements

Statement: A committed value satisfies some constraint.

LaTeX:

\text{Given } C = \text{Commit}(v), \text{ prove: } f(v) = 1

Example: Prove the committed location is within a valid region using a constraint on coordinates.

1.5 Correctness of Computation

Statement: A program P run on input x produces output y.

LaTeX:

\text{Prove: } \exists w \text{ such that } P(x, w) = y

Example: Prove that a batch of transactions was verified correctly using a zkRollup circuit.

1.6 Non-Disclosure Proofs

Statement: A property of a secret holds (e.g., age > 18).

LaTeX:

\text{Given } C = \text{Commit}(\text{age}), \text{ prove: } \text{age} > 18

Example: Prove you are eligible to access age-restricted content without disclosing your actual age.

1.7 Set Intersection

Statement: Two private sets share at least one element.

LaTeX:

\text{Prove: } \exists x \in S_1 \cap S_2

Example: Prove that two encrypted friend lists share at least one mutual contact without revealing the full lists.

2. Composite Zero-Knowledge Proofs

2.1 zk-KYC

Goal: Prove that:

You are on a KYC allowlist.

You are over 18.

LaTeX:

\text{Prove: } \text{MerkleVerify}(\text{hash}(id)) = 1 \land \text{age} \geq 18

Application: On-chain access control for age-restricted DAOs or mints.

2.2 zkRollup Transaction Validity

Goal: Prove all transactions in a batch are valid.

Sub-Statements:

Signature verification

Input = Output balance

Merkle tree root update

LaTeX:

\forall tx_i: \text{SigVerify}(tx_i) \land \text{ValidState}(tx_i) \land \text{UpdateRoot}(tx_i)

Application: Layer 2 scalability and fraud-proof execution.

2.3 zkCredential

Goal: Prove possession of a credential with attributes satisfying constraints.

LaTeX:

\exists id, age: \text{MerkleVerify}(\text{hash}(id)) = 1 \land \text{age} \geq 18 \land \text{country} = \text{US}

Application: Anonymous voting, regional access controls, decentralized identity.

3. Zero-Knowledge Machine Learning (zkML)

3.1 zkML Inference Proof

Goal: Prove the output of a model is correct without revealing inputs or weights.

LaTeX:

\exists x : \text{Model}(x) = y

Example: Prove a facial embedding matches a known identity vector.

3.2 zkML Fairness Proof

Goal: Prove a model is fair with respect to group A and B outcomes.

LaTeX:

|p_A - p_B| < \epsilon

Example: Prove a hiring model does not favor male over female candidates.

3.3 zkML Accuracy Proof

Goal: Prove your model has a high test accuracy without leaking the model weights.

LaTeX:

\text{Accuracy}(M, \mathcal{D}) > 90\%

Example: Publish verifiable benchmark results on-chain.

4. Composite zkML Circuits

4.1 Biometric Login + ID Check

Goal:

Input matches biometric classifier.

Belongs to approved identity group.

LaTeX:

\exists x, id : \text{Model}(x) = \text{valid} \land \text{MerkleVerify}(id) = 1

4.2 zkML + Credential Filter

Goal:

Model classifies resume as relevant.

Candidate has valid credential.

Age > 21

LaTeX:

\text{Model}(resume) = 1 \land \text{MerkleVerify}(cred) \land \text{age} > 21

Application: Private but verifiable job matching systems.

5. Hybrid zk + zkML Systems

5.1 zkML Guard for zkRollup

Goal: Verify zkRollup batch is fraud-free with zkML classifier.

LaTeX:

\text{zkProof}(\text{txs}) \land \text{AI}(\text{txs}) < \text{fraud\_threshold}

5.2 zkID + AI Moderation

Goal:

User is allowed.

Post is not toxic.

LaTeX:

\text{MerkleVerify}(id) \land \text{Toxicity}(msg) < 0.2

6. Summary Table

Layer

Purpose

Example

zkML

Private inference

Prove ML run was correct

zk

Constraint checking

ID, allowlist, balance

Composite zkML

Combine model + predicate

Resume screening, biometric login

Hybrid zk+ML

AI systems nested in zk logic

AI moderation, zkML + zkRollup

zk+Credential

Identity + private attribute

KYC, voting eligibility

zk+ML+Cred

All combined

ML screening with private ID + policy

