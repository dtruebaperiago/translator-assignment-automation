# Data Dictionary & Field Explanations

This document describes the structure and fields of the datasets used in the AssignMate decision support tool.

---

## Data Table

This table contains historical translation, proofreading, and engineering tasks, along with their assignment and execution metadata.

- **PROJECT_ID**: Project code (additional info, likely not necessary).
- **PM**: Responsible management team.
- **TASK_ID**: Unique identifier for the task.
- **START**: Task start date.
- **END**: Theoretical task delivery date / deadline (can be compared with the `DELIVERED` date to check for delays).
- **TASK_TYPE**: The nature of the task. Notable types include:
  - **Translation**: Core translation task. The translator's quality can be slightly lower if a high-quality ProofReading step is scheduled afterward.
  - **ProofReading**: Full linguistic review of a Translation or PostEditing task. This always follows the initial translation step and must be assigned to a translator with higher experience than the original translator.
  - **PostEditing**: Post-editing of machine-translated content. Similar to translation but requires specialized post-editing skills.
  - **Spotcheck**: Partial review of a Translation or PostEditing task. Requires higher experience, similar to ProofReading.
  - **DTP**: Desktop-Publishing / formatting tasks.
  - **Engineering**: Technical engineering tasks (file conversions, coding, preparation).
  - **LanguageLead**: Linguistic management and quality assurance tasks, assigned to highly experienced leads.
  - **Management**: General project management tasks.
  - **TEST**: Test translation required to qualify a new translator for a specific client. Assigned to the most experienced and high-quality translators.
  - **Training**: Internal training tasks (experience/quality requirements are not evaluated).
  - **Miscellaneous**: Various uncategorized linguistic tasks.
- **SOURCE_LANG**: Source language.
- **TARGET_LANG**: Target language.
- **TRANSLATOR**: Unique identifier / name of the translator.
- **ASSIGNED**: Timestamp when the task is pre-assigned to the translator.
- **READY**: Timestamp when the translator is notified they can begin working.
- **WORKING**: Timestamp when the translator starts working on the task.
- **DELIVERED**: Timestamp when the translator delivers the completed task.
- **RECEIVED**: Timestamp when the PM receives and approves the task.
- **CLOSE**: Timestamp when the PM marks the task as closed.
- **FORECAST**: Estimated hours needed for task completion.
- **HOURLY_RATE**: Hourly rate charged by the translator for the task.
- **COST**: Total cost of the task (usually computed as `FORECAST * HOURLY_RATE`).
- **QUALITY_EVALUATION**: Post-delivery quality score evaluated by proofreaders or PMs.
- **MANUFACTURER**: Client name.
- **MANUFACTURER_SECTOR**: Client sector (Level 1 category).
- **MANUFACTURER_INDUSTRY_GROUP**: Client industry group (Level 2 category).
- **MANUFACTURER_INDUSTRY**: Client industry (Level 3 category).
- **MANUFACTURER_SUBINDUSTRY**: Client sub-industry (Level 4 category).

---

## Schedules Table

This table models the availability of translators.

- **NAME**: Unique name of the translator.
- **START**: Start time of the translator's daily shift.
- **END**: End time of the translator's daily shift.
- **MON** / **TUES** / **WED** / **THURS** / **FRI** / **SAT** / **SUN**: Boolean flags (1 or 0) indicating whether the translator works on that weekday.

---

## Clients Table

This table contains client preferences and constraints.

- **CLIENT_NAME**: Unique name of the client.
- **SELLING_HOURLY_PRICE**: The hourly rate charged to the client.
- **MIN_QUALITY**: The minimum quality evaluation score required for tasks delivered to this client.
- **WILDCARD**: The SLA condition that can be relaxed when no perfect candidate is available (e.g., `Price`, `Quality`, or `Deadline`).

---

## Translators Cost + Pairs Table

This table lists active cost rates for specific language pairs.

- **TRANSLATOR**: Unique name of the translator.
- **SOURCE_LANG**: Source language supported.
- **TARGET_LANG**: Target language supported.
- **HOURLY_RATE**: Cost per hour charged for this language pair.

---

## Other Considerations

- **Experience Evaluation**: Translator experience is dynamically evaluated based on the number of hours they have historically worked for a specific client, industry sector, or task type.
