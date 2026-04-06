import os

req_path = '/Users/hvsingh/Desktop/form_eval_app/server/rag/requirements.txt'
with open(req_path, 'r') as f:
    lines = f.readlines()

with open(req_path, 'w') as f:
    for line in lines:
        if 'faiss-cpu' not in line:
            f.write(line)
