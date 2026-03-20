**iDISC AssignMate - Group Project**

**What is this project?**
This is our group's prototype for the iDISC challenge. 
We are building a decision-support tool to help their Project Managers (PMs) figure out the best translator for any given task.
Since iDISC has a ton of rules (schedules, language pairs, client quality demands), we decided against building a "black box" that assigns tasks automatically. 
Instead, we are building an AI "Copilot" dashboard. It will calculate all the constraints and suggest the top 3 best translators for the job, letting the PM make the final click.

**What we've done so far**

1. Design Thinking & ConceptWe used the 6-step Design Thinking framework to figure out what to actually build.
We spent time trying to empathize with the PMs  and realized they are wasting time manually cross-referencing four different tables.
During the ideation and prototyping phases , we came up with the "AssignMate Dashboard" concept so PMs can actually see why the AI is recommending someone.
   
2. Variable AnalysisWe went through the dataset and ranked the variables to figure out what our ML model should actually focus on:

Top Priority:
- The individual translator, their domain experience (we'll need to calculate this from past data),
- task dependencies (like making sure a Proofreader is more experienced than the Translator), and the client's minimum quality threshold.

Medium Priority: 
- Language pairs,
- industry types,
- time pressure (forecasted hours vs. actual deadlines).

Low/No Priority: 
- We are going to ignore administrative stuff like PROJECT_ID or Kanban timestamps (RECEIVED, CLOSE) since they don't impact the actual translation quality.

3. Our Coding StrategyWe figured out our pipeline.
We can't just throw everything into a basic machine learning model.

Data Split: 
We are going to use a chronological time-series split for our Train/Validation/Test sets. 
If we use a random split, our model will "cheat" by using future data to predict past assignments.

Hard Constraints:
First, we will use rule-based coding to filter out people who physically can't do the job (e.g., they don't speak the language or don't work on Fridays).

Soft Constraints:
Then, we will use an ML model to score and rank whoever is left based on predicted quality, punctuality, and cost.


**Next Steps** (Our To-Do List)

Step 1: Repo & Environment[ ] Set up this GitHub repo for everyone.[ ] Make sure we all have the same requirements.txt installed (pandas, scikit-learn, etc.) so our code doesn't break on different laptops.

Step 2: Data Preprocessing[ ] Load the four tables (Data Table, Schedules, Clients, TranslatorsCost+Pairs) into Pandas.[ ] Clean up the dates and missing values.[ ] Crucial: Create our synthetic variables. We need to write code that loops through the past data to calculate a running "Experience Score" for each translator based on the hours they've already worked for specific industries.

Step 3: Filtering (Hard Constraints)[ ] Write the logic to merge the task requirements with the Schedules and TranslatorsCost+Pairs tables.[ ] Write the code to drop anyone from the candidate pool who doesn't match the languages or is off the clock.

Step 4: The ML Model (Soft Constraints)[ ] Train a predictive model (probably starting with a Random Forest) on the training set to predict what QUALITY_EVALUATION score a translator will likely get.[ ] Write an objective function that takes that predicted quality, mixes it with the translator's hourly rate, and spits out our "Top 3" ranking.
