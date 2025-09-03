# START Project Documentation Hub

Welcome to the START (Scalable, Tailored Active-inference Research & Training) documentation.

- [Getting Started](getting_started.md)
- [Environment Setup](environment.md)
- [Pipeline Overview](pipeline.md)
- [Configuration](configuration.md)
- [Examples](examples.md)
- [Repository & Clone Management](clones.md)
- [Testing Guide](TESTING.md)

## Quick Links

- Learning Guides:
  - [Curriculum Creation Usage Guide](learning/curriculum_creation/USAGE_GUIDE.md)
  - [API Integration Guide](learning/curriculum_creation/README.md)
- Configuration Reference:
  - `data/config/entities.yaml`
  - `data/config/domains.yaml`
  - `data/config/languages.yaml`
- Prompt Templates:
  - `data/prompts/research_domain_analysis.md`
  - `data/prompts/research_domain_curriculum.md`
  - `data/prompts/research_entity.md`
  - `data/prompts/translation.md`

```mermaid
graph TD
  A["START Docs Hub"] --> B["Environment Setup"]
  A --> C["Pipeline Overview"]
  A --> D["Repository & Clone Management"]
  A --> E["Testing Guide"]
  A --> F["User & API Guides"]
  F --> F1["Curriculum Creation Usage Guide"]
  F --> F2["API Integration Guide"]
  A --> I["Getting Started"]
  A --> J["Configuration"]
  A --> K["Examples"]

  click B "environment/" "Environment Setup"
  click C "pipeline/" "Pipeline Overview"
  click D "clones/" "Repository & Clone Management"
  click E "TESTING/" "Testing Guide"
  click F1 "learning/curriculum_creation/USAGE_GUIDE/" "Usage Guide"
  click F2 "learning/curriculum_creation/README/" "API Integration Guide"
  click I "getting_started/" "Getting Started"
  click J "configuration/" "Configuration"
  click K "examples/" "Examples"
```

Fallback links:

- [Environment Setup](environment/)
- [Pipeline Overview](pipeline/)
- [Repository & Clone Management](clones/)
- [Testing Guide](TESTING/)
- [Curriculum Creation Usage Guide](learning/curriculum_creation/USAGE_GUIDE/)
- [API Integration Guide](learning/curriculum_creation/README/)
- [Getting Started](getting_started/)
- [Configuration](configuration/)
- [Examples](examples/)
