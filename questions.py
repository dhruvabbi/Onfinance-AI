import google.generativeai as genai
import instructor
from pydantic import BaseModel

def gemini(a):
    genai.configure(api_key="AIzaSyBAOzHFjPoXhIo5HXR3KWMqE4K8KY8xrmQ")
    model = genai.GenerativeModel('gemini-1.5-pro')
    response = model.generate_content(a)
    return response.text

sample="""
| **Company Name**     | **Ticker Symbol** | **Market Cap (Billion USD)** | **P/E Ratio** | **Dividend Yield (%)** | **Earnings Growth Rate (%)** | **Debt-to-Equity Ratio** | **Sector**               |
|----------------------|-------------------|-----------------------------|---------------|------------------------|-----------------------------|--------------------------|--------------------------|
| Apple Inc.           | AAPL              | 2,500                       | 28.5          | 0.6                    | 10.3                        | 1.5                      | Technology               |
| Microsoft Corp.      | MSFT              | 2,100                       | 35.2          | 0.8                    | 14.5                        | 0.7                      | Technology               |
| Amazon.com Inc.      | AMZN              | 1,600                       | 62.1          | 0                      | 20.2                        | 0.9                      | Consumer Discretionary   |
| Alphabet Inc.        | GOOGL             | 1,800                       | 30.1          | 0                      | 17.5                        | 0.3                      | Communication Services   |
| Tesla Inc.           | TSLA              | 900                         | 70.4          | 0                      | 25.0                        | 1.0                      | Consumer Discretionary   |
| Johnson & Johnson    | JNJ               | 420                         | 22.4          | 2.6                    | 6.8                         | 0.4                      | Healthcare               |
| JPMorgan Chase & Co. | JPM               | 490                         | 13.7          | 2.3                    | 8.7                         | 1.2                      | Financials               |
| Procter & Gamble Co. | PG                | 340                         | 24.3          | 2.4                    | 4.5                         | 0.5                      | Consumer Staples        |
| Nvidia Corp.         | NVDA              | 600                         | 50.2          | 0.1                    | 22.3                        | 0.4                      | Technology               |
| Visa Inc.            | V                 | 500                         | 32.6          | 0.6                    | 13.8                        | 0.6                      | Financials               |
"""

input_table ="""
| Maturity (Years) | Duration | Convexity | YTM (%) | Macaulay Duration (Years) |
|------------------|----------|-----------|---------|--------------------------|
| 1                | 0.98     | 0.05      | 4.50    | 0.98                     |
| 3                | 2.94     | 0.20      | 5.25    | 2.88                     |
| 5                | 4.90     | 0.38      | 5.75    | 4.76                     |
| 7                | 6.86     | 0.65      | 6.10    | 6.57                     |
| 10               | 9.80     | 1.10      | 6.50    | 9.38                     |
"""

example_prompt=sample+"\nThis is an example table. A few example analytical questions based on the table are: \n1.Compare and contrast Amazon and Procter & Gamble based on their financials.\n2.Which companies will appeal to investors seeking high potential returns despite higher risk?\n\n"
final_prompt = example_prompt+"Based on the below table can you create some analytical questions. You can use the sample table and sample questions as an example. Retrun the questions in a list format without any extra headings."+input_table
result=gemini(final_prompt)

class User(BaseModel):
    questions: list[str]

client = instructor.from_gemini(
    client=genai.GenerativeModel(
        model_name="models/gemini-1.5-pro",
    ),
    mode=instructor.Mode.GEMINI_JSON,
)

resp = client.messages.create(
    messages=[
        {
            "role": "user",
            "content": result,
        }
    ],
    response_model=User,
)

print(resp.questions)
