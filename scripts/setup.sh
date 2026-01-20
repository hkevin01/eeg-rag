#!/bin/bash
# EEG-RAG Quick Setup Script
# Works on Linux, macOS, and WSL
# Usage: curl -sSL https://raw.githubusercontent.com/hkevin01/eeg-rag/main/scripts/setup.sh | bash

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                                                                  ║"
echo "║   ███████╗███████╗ ██████╗       ██████╗  █████╗  ██████╗       ║"
echo "║   ██╔════╝██╔════╝██╔════╝       ██╔══██╗██╔══██╗██╔════╝       ║"
echo "║   █████╗  █████╗  ██║  ███╗█████╗██████╔╝███████║██║  ███╗      ║"
echo "║   ██╔══╝  ██╔══╝  ██║   ██║╚════╝██╔══██╗██╔══██║██║   ██║      ║"
echo "║   ███████╗███████╗╚██████╔╝      ██║  ██║██║  ██║╚██████╔╝      ║"
echo "║   ╚══════╝╚══════╝ ╚═════╝       ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝       ║"
echo "║                                                                  ║"
echo "║          RAG System for EEG Research Literature                  ║"
echo "║                                                                  ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check for required tools
check_requirements() {
    echo -e "${YELLOW}Checking requirements...${NC}"
    
    # Check Python
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
        
        if [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 9 ]; then
            echo -e "  ${GREEN}✓${NC} Python $PYTHON_VERSION"
        else
            echo -e "  ${RED}✗${NC} Python 3.9+ required (found $PYTHON_VERSION)"
            exit 1
        fi
    else
        echo -e "  ${RED}✗${NC} Python 3 not found"
        echo "    Install Python 3.9+ from https://www.python.org/downloads/"
        exit 1
    fi
    
    # Check pip
    if command -v pip3 &> /dev/null || command -v pip &> /dev/null; then
        echo -e "  ${GREEN}✓${NC} pip available"
    else
        echo -e "  ${RED}✗${NC} pip not found"
        exit 1
    fi
    
    # Check git (optional but recommended)
    if command -v git &> /dev/null; then
        echo -e "  ${GREEN}✓${NC} git available"
        HAS_GIT=true
    else
        echo -e "  ${YELLOW}!${NC} git not found (optional)"
        HAS_GIT=false
    fi
    
    # Check Docker (optional)
    if command -v docker &> /dev/null; then
        echo -e "  ${GREEN}✓${NC} Docker available (optional)"
        HAS_DOCKER=true
    else
        echo -e "  ${YELLOW}!${NC} Docker not found (optional, for containerized deployment)"
        HAS_DOCKER=false
    fi
}

# Create virtual environment
create_venv() {
    echo ""
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    
    if [ -d "venv" ]; then
        echo "  Virtual environment already exists"
    else
        python3 -m venv venv
        echo -e "  ${GREEN}✓${NC} Virtual environment created"
    fi
    
    # Activate venv
    source venv/bin/activate
    echo -e "  ${GREEN}✓${NC} Virtual environment activated"
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel -q
    echo -e "  ${GREEN}✓${NC} pip upgraded"
}

# Install EEG-RAG
install_eeg_rag() {
    echo ""
    echo -e "${YELLOW}Installing EEG-RAG...${NC}"
    
    pip install -e . -q
    echo -e "  ${GREEN}✓${NC} EEG-RAG installed"
}

# Create data directories
setup_directories() {
    echo ""
    echo -e "${YELLOW}Setting up directories...${NC}"
    
    mkdir -p data/raw data/processed data/embeddings/cache logs
    echo -e "  ${GREEN}✓${NC} Data directories created"
}

# Create .env file
create_env_file() {
    echo ""
    echo -e "${YELLOW}Creating configuration...${NC}"
    
    if [ -f ".env" ]; then
        echo "  .env file already exists"
    else
        cat > .env << 'EOF'
# EEG-RAG Configuration
# See docs/configuration.md for all options

# OpenAI API Key (required for LLM features)
# Get your key at: https://platform.openai.com/api-keys
OPENAI_API_KEY=your-api-key-here

# PubMed Email (recommended for API access)
PUBMED_EMAIL=your-email@example.com

# Optional: Neo4j (for knowledge graph features)
# NEO4J_URI=bolt://localhost:7687
# NEO4J_USER=neo4j
# NEO4J_PASSWORD=your-password

# Optional: Redis (for caching)
# REDIS_URL=redis://localhost:6379
EOF
        echo -e "  ${GREEN}✓${NC} .env file created"
        echo -e "  ${YELLOW}!${NC} Remember to add your OPENAI_API_KEY to .env"
    fi
}

# Run verification
verify_installation() {
    echo ""
    echo -e "${YELLOW}Verifying installation...${NC}"
    
    python3 -c "import eeg_rag; print(f'  EEG-RAG version: {eeg_rag.__version__}')" 2>/dev/null || \
    python3 -c "import eeg_rag; print('  EEG-RAG imported successfully')"
    
    echo -e "  ${GREEN}✓${NC} Installation verified"
}

# Install Ollama and Mistral for local LLM
install_ollama() {
    echo ""
    echo -e "${YELLOW}Installing Ollama for local LLM (Mistral)...${NC}"
    
    if command -v ollama &> /dev/null; then
        echo -e "  ${GREEN}✓${NC} Ollama already installed"
    else
        echo "  Downloading and installing Ollama..."
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            curl -fsSL https://ollama.com/install.sh | sh
        elif [[ "$OSTYPE" == "darwin"* ]]; then
            echo "  Please install Ollama manually from: https://ollama.com/download"
            echo "  Or use: brew install ollama"
        else
            echo -e "  ${YELLOW}!${NC} Unsupported OS for automatic Ollama install"
            echo "  Please install manually from: https://ollama.com/download"
            return
        fi
        echo -e "  ${GREEN}✓${NC} Ollama installed"
    fi
    
    # Start Ollama service
    if pgrep -x "ollama" > /dev/null; then
        echo -e "  ${GREEN}✓${NC} Ollama service is running"
    else
        echo "  Starting Ollama service..."
        ollama serve > /dev/null 2>&1 &
        sleep 2
        echo -e "  ${GREEN}✓${NC} Ollama service started"
    fi
    
    # Pull Mistral model
    echo "  Downloading Mistral model (this may take a few minutes)..."
    ollama pull mistral
    echo -e "  ${GREEN}✓${NC} Mistral model ready"
}

# Print next steps
print_next_steps() {
    echo ""
    echo -e "${GREEN}╔══════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                    Installation Complete! 🎉                      ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "Next steps:"
    echo ""
    echo "  1. Activate the virtual environment:"
    echo -e "     ${BLUE}source venv/bin/activate${NC}"
    echo ""
    echo "  2. Start the web GUI with Mistral AI:"
    echo -e "     ${BLUE}streamlit run src/eeg_rag/web_ui/app.py${NC}"
    echo ""
    echo "  3. Or run the demo:"
    echo -e "     ${BLUE}python examples/demo_all_components.py${NC}"
    echo ""
    echo "  4. Run tests:"
    echo -e "     ${BLUE}make test${NC}"
    echo ""
    echo "For Docker deployment:"
    echo -e "     ${BLUE}make docker-up-lite${NC}    # Quick start (no external deps)"
    echo -e "     ${BLUE}make docker-up${NC}         # Full stack with Neo4j + Redis"
    echo ""
    echo -e "${BLUE}✨ EEG-RAG now uses Mistral AI locally (no API key needed!)${NC}"
    echo ""
    echo "Documentation: https://github.com/hkevin01/eeg-rag#readme"
    echo ""
}

# Main execution
main() {
    check_requirements
    create_venv
    install_eeg_rag
    setup_directories
    create_env_file
    install_ollama
    verify_installation
    print_next_steps
}

# Run if not sourced
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main
fi
