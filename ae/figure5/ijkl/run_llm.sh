cd ../../..

# Modular transformer block experiments (MHA vs GQA) on A100

# Auto-regressive, simulated A100, MHA
python -m ae.figure5.ijkl.test_llm --simgpu --attention-kind mha

# Auto-regressive, simulated A100, GQA with num_kv_heads=8
python -m ae.figure5.ijkl.test_llm --simgpu --attention-kind gqa --num-kv-heads 8

cd ae/figure5/ijkl
python plot_llm_gqa.py
