install:
    #!/bin/bash
    set -exo pipefail
    
    command -v ollama || curl -fsSL https://ollama.com/install.sh | sh
    ollama pull llama3

    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install -r requirements.txt
    pip install flash-attn --no-build-isolation   
    

    





