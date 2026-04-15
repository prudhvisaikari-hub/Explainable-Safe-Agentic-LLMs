# Research Paper Images: Explainable Safe Agentic LLMs

## Figure 1: System Architecture Overview

```mermaid
graph TB
    UI["User Interaction Layer<br/>(Goals, Constraints, Permissions)"]
    PL["Planner-Reasoner Layer<br/>(Task Decomposition, ReAct)"]
    SP["Safety & Policy Layer<br/>(Governance, Risk Assessment)"]
    TE["Tool Execution Layer<br/>(Sandbox, Capability Bounds)"]
    OB["Observability & Explanation<br/>(Traces, Audits)"]
    
    UI -->|Structured Tasks| PL
    PL -->|Planned Actions| SP
    SP -->|Validated Actions| TE
    TE -->|Execution Results| OB
    OB -->|Feedback Loop| UI
    OB -->|Policy Updates| SP
    
    style UI fill:#e1f5ff
    style PL fill:#f3e5f5
    style SP fill:#ffe0b2
    style TE fill:#e8f5e9
    style OB fill:#fce4ec
```

## Figure 2: Safety & Policy Control Architecture

```mermaid
graph LR
    Input["External Input<br/>(User/Web/Memory)"]
    PInjection["Prompt Injection<br/>Detection"]
    PolicyCheck["Policy Validator<br/>(Tool-Specific Rules)"]
    RiskScore["Risk Score<br/>Computation"]
    Decision{"Action<br/>Allowed?"}
    
    Execute["✓ Execute Action<br/>with Metadata"]
    Block["✗ Block Action<br/>Log Violation"]
    Review["⚠ Human Review<br/>Required"]
    
    Input --> PInjection
    PInjection --> PolicyCheck
    PolicyCheck --> RiskScore
    RiskScore --> Decision
    
    Decision -->|Low Risk| Execute
    Decision -->|Blocked| Block
    Decision -->|High Risk| Review
    
    Execute --> Log["Audit Log"]
    Block --> Log
    Review --> Log
    
    style Input fill:#fff3e0
    style PInjection fill:#ffcdd2
    style PolicyCheck fill:#fff9c4
    style RiskScore fill:#c8e6c9
    style Execute fill:#a5d6a7
    style Block fill:#ef5350
    style Review fill:#ffb74d
```

## Figure 3: Explainability Pipeline

```mermaid
graph TB
    Exec["Agent Execution"]
    
    DT["Decision Trace<br/>Capture"]
    ER["Evidence Retrieval<br/>Linking"]
    CF["Counterfactual<br/>Analysis"]
    
    NL["Natural Language<br/>Generation"]
    Struct["Structured<br/>Format"]
    Visual["Visualization<br/>Dashboard"]
    
    Exec --> DT
    Exec --> ER
    Exec --> CF
    
    DT --> NL
    ER --> Struct
    CF --> Visual
    
    NL --> Output["User-Facing<br/>Explanation"]
    Struct --> Output
    Visual --> Output
    
    style Exec fill:#e3f2fd
    style DT fill:#f1f8e9
    style ER fill:#ede7f6
    style CF fill:#fce4ec
    style Output fill:#c8e6c9
```

## Figure 4: Multi-Agent Architecture

```mermaid
graph TB
    Central["Central Planner<br/>(Task Decomposition)"]
    
    Safe["SafeAgent<br/>(Risk-Aware RL)"]
    Human["HumanAgent<br/>(Oversight)"]
    Specialist["SpecialistAgent<br/>(Domain-Specific)"]
    
    Memory["Shared Memory<br/>(Context & History)"]
    Safety["Safety Policy<br/>Constraints"]
    
    Central --> Safe
    Central --> Human
    Central --> Specialist
    
    Safe --> Memory
    Human --> Memory
    Specialist --> Memory
    
    Memory --> Safety
    Safety --> Central
    
    style Central fill:#b3e5fc
    style Safe fill:#c8e6c9
    style Human fill:#ffe0b2
    style Specialist fill:#f8bbd0
    style Memory fill:#e1bee7
    style Safety fill:#ffccbc
```

## Figure 5: Evaluation Methodology

```mermaid
graph TB
    Framework["Explainable Safe Agentic LLM<br/>Framework"]
    
    Functional["Functional Metrics<br/>• Task Completion<br/>• Step Efficiency<br/>• Error Recovery"]
    
    Explainability["Explainability Metrics<br/>• Trace Completeness<br/>• Evidence Alignment<br/>• User Comprehension<br/>• Explanation Fidelity"]
    
    Safety["Safety Metrics<br/>• Policy Compliance<br/>• Prompt Injection Resistance<br/>• Tool Misuse Prevention<br/>• False Allow/Block Rates"]
    
    Observability["Observability Metrics<br/>• Audit Trace Quality<br/>• Event Reconstruction<br/>• Safety Rule Verification"]
    
    Framework --> Functional
    Framework --> Explainability
    Framework --> Safety
    Framework --> Observability
    
    Functional --> Results["Comprehensive<br/>Evaluation Results"]
    Explainability --> Results
    Safety --> Results
    Observability --> Results
    
    style Framework fill:#bbdefb
    style Functional fill:#a5d6a7
    style Explainability fill:#ffe082
    style Safety fill:#ef9a9a
    style Observability fill:#ce93d8
    style Results fill:#90caf9
```

## Figure 6: Component Interaction Flow

```mermaid
graph LR
    Input["1. User Input"]
    Normalize["2. Normalize &<br/>Tag Risk"]
    Plan["3. Plan Actions"]
    Validate["4. Safety Check"]
    Execute["5. Execute with<br/>Sandbox"]
    Trace["6. Capture Trace"]
    Explain["7. Generate<br/>Explanation"]
    
    Input --> Normalize
    Normalize --> Plan
    Plan --> Validate
    Validate -->|Pass| Execute
    Validate -->|Fail| Trace
    Execute --> Trace
    Trace --> Explain
    
    style Input fill:#ffecb3
    style Normalize fill:#ffe0b2
    style Plan fill:#ffccbc
    style Validate fill:#ffab91
    style Execute fill:#ef9a9a
    style Trace fill:#f48fb1
    style Explain fill:#ce93d8
```

## Figure 7: Explainability Methods Matrix

| Method | Description | Used For |
|--------|-------------|----------|
| **Decision Trace** | Step-by-step execution log | Understanding agent reasoning |
| **Evidence Linking** | Maps actions to source data | Accountability & auditability |
| **Counterfactual Explanation** | Why actions were blocked | Understanding boundaries |
| **Attention Analysis** | Which inputs influenced output | Interpretability |
| **Feature Importance** | Which features drove decision | Model understanding |
| **Natural Language Summary** | Human-readable rationale | User communication |

## Figure 8: Safety Threat Model

```mermaid
graph TB
    Threats["Security Threats"]
    
    Direct["Direct Prompt<br/>Injection"]
    Indirect["Indirect Prompt<br/>Injection"]
    Memory["Memory<br/>Contamination"]
    Tool["Tool Misuse"]
    
    Threats --> Direct
    Threats --> Indirect
    Threats --> Memory
    Threats --> Tool
    
    Direct --> Mitig1["Input Validation<br/>& Sandboxing"]
    Indirect --> Mitig2["Content Tagging<br/>& Trust Boundaries"]
    Memory --> Mitig3["Isolation & Access<br/>Controls"]
    Tool --> Mitig4["Least-Privilege<br/>Permissions"]
    
    Mitig1 --> Defense["Layered Defense<br/>Strategy"]
    Mitig2 --> Defense
    Mitig3 --> Defense
    Mitig4 --> Defense
    
    style Threats fill:#ffcdd2
    style Direct fill:#ef9a9a
    style Indirect fill:#ef9a9a
    style Memory fill:#ef9a9a
    style Tool fill:#ef9a9a
    style Defense fill:#a5d6a7
```

---

## How to Use These Diagrams

These diagrams are rendered using Mermaid and can be:
1. Viewed directly in GitHub markdown files
2. Exported as SVG/PNG using Mermaid CLI
3. Embedded in research papers and presentations
4. Modified to reflect specific implementation details

For use in Word or LaTeX documents:
- Right-click and save as image
- Use Mermaid CLI: `mmdc -i DIAGRAM.md -o diagram.svg`
- Take screenshots for direct inclusion

---

## Diagram Descriptions for Paper

### Figure 1: System Architecture
Illustrates the five-layer approach to Explainable Safe Agentic LLMs: user interaction, planning, safety governance, tool execution, and observability.

### Figure 2: Safety Control
Shows how the safety and policy layer implements multiple control gates, including prompt injection detection, policy validation, and risk scoring.

### Figure 3: Explainability
Demonstrates how decision traces, evidence retrieval, and counterfactual analysis are combined to produce user-facing explanations in natural language and structured formats.

### Figure 4: Multi-Agent Design
Illustrates the coordination of SafeAgent (risk-aware), HumanAgent (oversight), and SpecialistAgent (domain-specific) under a central planner with shared memory and safety constraints.

### Figure 5: Evaluation Framework
Shows the four dimensions of evaluation: functional metrics, explainability metrics, safety metrics, and observability metrics.

### Figure 6: Component Flow
Sequential diagram showing how user input flows through normalization, planning, safety validation, execution, tracing, and explanation generation.

### Figure 7: Explainability Methods
Table summarizing the different explanation methods and their purposes in the framework.

### Figure 8: Threat Model
Maps security threats (direct/indirect prompt injection, memory contamination, tool misuse) to mitigation strategies, emphasizing layered defense.
