# Experiment 06: Amplification on Gemma 4

## Goal
Test if activation amplification can surface hidden reasoning in Gemma 4.

## Key Questions
1. Does amplification reveal suppressed information?
2. Which layers are most sensitive?
3. What's the sweet spot for coherent but revealing outputs?

## Method
1. Extract "hint direction" = mean(hint_activations) - mean(no_hint_activations)
2. During generation, amplify activations along this direction
3. Compare baseline vs amplified outputs

## Experiments
- **Layer sweep**: Test amplification at layers [0, 8, 17, 26, 33]
- **Factor sweep**: Test factors [0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0]

## Usage
```bash
cd ~/jain && source .venv/bin/activate
python experiments/06_gemma4_amplify/scripts/amplify_gemma4.py --amplify 1.2
```
