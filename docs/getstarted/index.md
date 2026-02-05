# Get Started with TempoEval

Welcome to the **TempoEval** documentation. 

This section guides you through setting up the framework and running your first evaluation.

<div class="grid cards" markdown>

-   :material-download: **[Installation](installation.md)**
    
    Detailed guide on installing via pip, setting up Java for HeidelTime, and configuring environment variables.

-   :material-rocket-launch: **[Quick Start](quickstart.md)**
    
    Run your "Hello World" evaluation in 5 minutes. Learn the core loop: Extract -> Compute.

</div>

## Why use TempoEval?

Most RAG metrics (like Ragas or TruLens) focus on **semantic context**. TempoEval focuses specifically on **time**.

!!! example "The Temporal Distinction"
    **Query**: *"Who was president in 1999?"*

    - **President Clinton** (1993-2001) ✅ Temporally correct
    - **President Bush** (2001-2009) ❌ Temporally incorrect

    Both are semantically similar (both presidents), but **temporally distinct**. TempoEval catches this.

### How Focus Time Works

**The Process:**

1. **Extract Query Focus Time (QFT)**: Query *"What caused the 2008 financial crisis?"* → `{2008}`
2. **Extract Document Focus Times (DFT)**:
   - Doc 1: *"Lehman Brothers collapsed in 2008..."* → `{2008}` ✅
   - Doc 2: *"The 1929 Wall Street Crash..."* → `{1929}` ❌
   - Doc 3: *"COVID-19 pandemic in 2020..."* → `{2020}` ❌
3. **Compare**: Match QFT with each DFT using set intersection
4. **Score**: Temporal Precision = 1/3 = 33% (only Doc 1 matches)
