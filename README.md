# Speech-SSL-FFN-Analysis

## Example of Generate Property-Specific Keys for Gender 
1. Copy modified files in s3prl/upstream into original s3prl
2. Generate property-specific keys of pretrained HuBERT base for gender 
  ```
  bash generate.sh [MFA Directory] [Librispeech Directory] [Processing Stage]
  bash generate.sh /home/nervjack2/Desktop/dataset/LibriSpeechMFA/dev-clean/ /home/nervjack2/Desktop/dataset/LibriSpeech 0
  ```
