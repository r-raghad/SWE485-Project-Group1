# Executive Credit Risk Briefing

## 1. Risk Summary
The supervised machine learning model predicts a status of **Default** for this customer, with an individual **Default Probability of 54.72%**. 

Conversely, the unsupervised clustering model places this customer into the **"Active On-Time Payers"** segment, a group characterized by a **Low** risk level and a historical **Segment Default Rate of 13.8%**. 

Because the individual's predicted default probability (54.72%) is substantially higher than the historical baseline of their assigned segment (13.8%), this account presents a notable divergence that requires targeted review.

---

## 2. Key Risk Factors
* **High Individual Default Probability:** The supervised model evaluates the customer's specific risk of default at 54.72%, triggering a formal "Default" prediction.
* **Model Divergence:** The individual's elevated risk score conflicts directly with the broader segment's historically low default rate (13.8%). This suggests that specific individual features or recent behavioral changes—separate from general segment trends—are driving high risk signals.

---

## 3. Customer Segment Interpretation
* **Segment Profile:** The customer belongs to the *Active On-Time Payers* cluster. As a group, this segment consists of clients who use their cards regularly, maintain above-average bills and payment amounts, and average 0.2 late months out of 6. The segment's aggregate traits include a median credit limit of NT$150,000 and a median average bill of NT$33,790.
* **Standard Segment Strategy:** The default policy for this segment is "Retain and grow," which includes eligibility for credit-limit increases, rewards, cross-selling, and standard monitoring.
* **Analytical Nuance & Limitations:** 
  * *Individual vs. Segment Metrics:* The segment default rate (13.8%) reflects historical group performance, whereas the 54.72% probability is tailored to this specific individual. 
  * *Trait Attribution:* The aggregate segment characteristics (such as median credit limit or average bill size) describe the cluster as a whole and must not be assumed to reflect this specific customer's actual account balance or credit limit.

---

## 4. Recommended Actions (For Human Review)
The following recommendations are provided to guide credit officers and underwriters in performing a manual evaluation; they do not constitute a final lending decision.

1. **Conduct Manual Underwriting Review:** Pause automatic segment-based actions (such as credit-limit increases, rewards, or cross-selling initiatives) until a credit risk officer manually reviews the file.
2. **Investigate Risk Divergence:** Examine the specific account-level attributes driving the supervised model’s 54.72% default prediction to identify underlying risk factors not captured by the segment classification.
3. **Adjust Monitoring Level:** Temporarily elevate account monitoring beyond the segment's standard protocol until the human review is completed and a formal credit decision is rendered.