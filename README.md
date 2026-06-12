# ATR Framework

Image editing framework with intelligent pipeline routing.

## Installation

```bash
cd ATR_Framework
pip install -r requirements.txt
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json"
```

## Quick Start

### Single Test

```bash
# Run with default agent (Qwen)
python scripts/quick_test.py

# Run with specific agent
python scripts/quick_test.py --agent qwen       # Qwen only
python scripts/quick_test.py --agent banana     # Banana only
python scripts/quick_test.py --agent both       # Both agents
```

### Batch Processing

```bash
# Run all cases from test.jsonl
python scripts/batch_run.py --agent both        # Default: both agents
python scripts/batch_run.py --agent qwen        # Qwen only
python scripts/batch_run.py --agent banana      # Banana only
python scripts/batch_run.py --agent both --max-samples 10
```

### One-Click Edit

```bash
# Basic usage
python scripts/run_edit.py --image photo.jpg --instruction "Change the sky to blue"

# With JSON file
python scripts/run_edit.py --json-file test.json --agent qwen

# With custom output
python scripts/run_edit.py --image photo.jpg --instruction "..." --output ./results
```

## Project Structure

```
ATR_Framework/
├── core/              # Core modules
├── tools/             # Editing tools
├── prompts_qwen/      # Qwen prompts
├── prompts_banana/    # Banana prompts
├── results/           # Output directory
├── scripts/           # Main scripts
│   ├── quick_test.py
│   ├── batch_run.py
│   └── run_edit.py
└── test.jsonl         # Test data
```

## Output

Results are saved in `results/{agent_type}/{index}/`:
- `caption.json` - Image analysis
- `routing.json` - Pipeline routing decision
- `trace.json` - Execution trace
- `report.json` - Complete report
- `output.jpg` - Final edited image
```



## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## Support

For issues and questions:
- 📧 Email: support@example.com
- 🐛 Issues: GitHub Issues
- 📚 Documentation: See docs/ folder

---

**Made with ❤️ for image editing**
