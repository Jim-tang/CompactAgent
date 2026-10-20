Your task is to create a detailed summary of the conversation so far, paying close attention to the user's explicit requests and your previous actions.
This summary should be thorough in capturing technical details, code patterns, and architectural decisions that would be essential for continuing development work without losing context.

### Required Summary Sections

- Primary Request and Intent — All explicit user requests and intents
- Key Technical Concepts — Technologies, frameworks, patterns discussed
- Files and Code Sections — Files examined/modified/created with full code snippets
- Errors and Fixes — Errors encountered and resolution steps
- Problem-Solving — Solved problems and ongoing troubleshooting
- All User Messages — Every non-tool-result user message (critical for tracking intent drift)
- Pending Tasks — Outstanding work items
- Current Work — Precise description of work in progress before summarization
- Optional Next Step — Next step only if directly aligned with recent user requests

Before providing your final summary, wrap your analysis in <analysis> tags to organize your thoughts and ensure you've covered all necessary points. In your analysis process:

1. Chronologically analyze each message and section of the conversation. For each section thoroughly identify:
   - The user's explicit requests and intents
   - Your approach to addressing the user's requests
   - Key decisions, technical concepts and code patterns
   - Specific details like:
     - file names
     - full code snippets
     - function signatures
     - file edits
   - Errors that you ran into and how you fixed them
   - Pay special attention to specific user feedback that you received, especially if the user told you to do something differently.

2. Double-check for technical accuracy and completeness, addressing each required element thoroughly.

CRITICAL: Respond with TEXT ONLY. Do NOT call any tools.
- You already have all the context you need in the conversation above. 
- Tool calls will be REJECTED and will waste your only turn — you will fail the task.
- Your entire response must be plain text: an <analysis> block followed by a <summary> block.