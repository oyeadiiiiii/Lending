Lending Loan Approval — Deep Learning vs Offline RL

This project compares two different approaches for making loan approval decisions using the LendingClub dataset.
One approach uses a Deep Learning model to predict loan default probability, and the other uses an Offline Reinforcement Learning (RL) agent that tries to directly maximize profit. The whole idea was to see which one actually ends up making better decisions from a business point of view.






Project Overview:
Loan approval systems usually use supervised learning to predict default risk, but that doesn’t always align with profit. For example, a high-interest loan might be risky but still profitable overall.
So the project tries both:
Deep Learning (DL)
-Standard neural network classifier
-Predicts probability that a borrower defaults
-Uses AUC and F1-score for evaluation
-Policy is basically “approve if risk < threshold”

Offline Reinforcement Learning (RL)
-Learned using Discrete CQL from the d3rlpy library
-Uses historical decisions as offline data
-Does not try to predict default, but tries to maximize long-term expected reward
-Reward = financial gain or loss depending on repayment







File Structure
LENDINGCLUB/
│
├── accepted_2007_to_2018Q4.csv        
├── preprocessing.ipynb                
├── d3rlpy_logs/                     
│
├── LendingClub FINAL REPORT .pdf      
├── requirements.txt
├── README.md
├── .gitignore
└── .gitattributes
The CSV file itself is excluded from git since it’s huge.





Data Preparation:
The main dataset includes borrower financial features like annual income, DTI, FICO score, loan amount, interest rate, etc.
I cleaned the dataset inside preprocessing.ipynb and turned everything into numeric values, dropped missing columns, and made the target:
-0 → fully paid
-1 → default

For the RL agent, each loan is considered one step.
Reward function:
-approved & fully paid: loan_amount * interest_rate
-approved & defaulted: -loan_amount
-denied: 0
This is obviously simplified but works for comparing policies.






Model Results:
Deep Learning (DL)
Metric	    Score
AUC	        0.7099
F1-score	  0.3124
The AUC is decent considering the data imbalance, and F1 is not great but typical for default prediction problems.

Reinforcement Learning (RL):
Metric	                            Value
Estimated Policy Value (EPV)	      166700.2
EPV basically means that if we used the RL policy, the model estimates around 166k profit based on historical data and rewards.





Key Findings
The two models don't really optimize the same thing:
-The DL model tries to classify defaults correctly.
-The RL model tries to approve loans that will make the most money overall.
So sometimes RL approves loans that DL would reject.-
For example, if someone has medium risk but a very high interest rate, the RL agent might approve because the expected profit outweighs the losses.
This makes the RL model actually act in a more “business-aware” way, while DL is purely risk-based.




Conclusion:
The DL model gives stable and interpretable risk ranking.
But the RL model achieves much higher expected profitability.
Still, deploying RL directly could be risky since offline RL depends a lot on how rewards are designed and on the quality of historical actions.
If I had to choose right now, I’d deploy the DL model and use the RL model more as a research or simulation tool.





Future Work:
A few things that would help improve the models:
Add macroeconomic data (unemployment rate, inflation, recession periods)
Include more borrower behavior history (monthly repayments, spending data)
Try better tabular models like LightGBM or TabNet
Use risk-aware reinforcement learning (distributional RL, CVaR RL)
Model multi-step rewards, not just single-step loan outcomes
There’s still a lot to explore here.



Installation:
pip install -r requirements.txt




Running the Code:
Open the preprocessing notebook and run all cells
Train the deep learning classifier
Train the CQL agent
Compare their decision outputs and profitability
