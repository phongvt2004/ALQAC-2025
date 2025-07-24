# Legal Question Answering with Chain-of-Thought Reasoning

This project details the process of developing and training a language model to answer legal questions across three formats: True/False, Multiple-choice, and Free-text. The core of our approach is to enable the model to generate a detailed, step-by-step reasoning process (Chain-of-Thought) before providing a final answer.

## Methodology

Our workflow is divided into three main stages: Data Augmentation, Model Training (SFT and GPRO), and Ensemble for Submission.

### 1. Data Preparation: Generating High-Quality "Think" Data

The initial dataset provided by the organizers contained questions and corresponding answers but lacked the reasoning process behind them. To create a high-quality dataset for Chain-of-Thought (CoT) fine-tuning, we augmented the data by generating a "think" component for each sample.

- **Model Used:** `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`
- **Objective:** To generate a structured reasoning process explaining *why* a particular answer is correct based on the provided context.

#### Example Data Generation Prompt (for True/False questions)

This prompt was used to guide the DeepSeek model in generating the detailed reasoning.

```  
Kết luận cuối cùng sau khi suy luận đã có, hãy phân tích để giải thích cho kết luận: 'Đúng' với câu hỏi dạng Đúng/Sai.

Dựa vào bối cảnh bên dưới, hãy phân tích kỹ trước khi trả lời câu hỏi.

Loại câu hỏi: Đúng/Sai 

Câu hỏi: Người nghiện ma túy từ đủ 18 tuổi trở lên bị áp dụng biện pháp xử lý hành chính đưa vào cơ sở cai nghiện bắt buộc theo quy định của Luật Xử lý vi phạm hành chính khi bị phát hiện sử dụng chất ma túy một cách trái phép trong thời gian cai nghiện ma túy tự nguyện, đúng hay sai?

Bối cảnh: 
Đối tượng bị áp dụng biện pháp xử lý hành chính đưa vào cơ sở cai nghiện bắt buộc

Người nghiện ma túy từ đủ 18 tuổi trở lên bị áp dụng biện pháp xử lý hành chính đưa vào cơ sở cai nghiện bắt buộc theo quy định của Luật Xử lý vi phạm hành chính khi thuộc một trong các trường hợp sau đây:

1. Không đăng ký, không thực hiện hoặc tự ý chấm dứt cai nghiện ma túy tự nguyện;

2. Trong thời gian cai nghiện ma túy tự nguyện bị phát hiện sử dụng trái phép chất ma túy;

3. Người nghiện ma túy các chất dạng thuốc phiện không đăng ký, không thực hiện hoặc tự ý chấm dứt điều trị nghiện các chất dạng thuốc phiện bằng thuốc thay thế hoặc bị chấm dứt điều trị nghiện các chất dạng thuốc phiện bằng thuốc thay thế do vi phạm quy định về điều trị nghiện;

4. Trong thời gian quản lý sau cai nghiện ma túy mà tái nghiện.

Hãy đưa ra câu trả lời theo format sau:
1. Phân tích câu hỏi: [trình bày ngắn gọn nội dung và ý định của câu hỏi]
2. Dẫn chứng từ bối cảnh:
   - Hãy tách từng đoạn dài trong bối cảnh thành nhiều ý nhỏ rõ ràng.
   - Hãy tách ít nhất 3 đến 5 ý ở phần dẫn chứng từ bối cảnh
   - Mỗi ý nên nêu rõ nội dung pháp lý, viết ngắn gọn dễ hiểu.
   - Ví dụ:
       - [ý 1 từ đoạn luật A]
       - [ý 2 từ đoạn luật A]
       - [ý 3 từ đoạn luật B]
   - Ghi rõ đoạn nào có liên quan đến câu hỏi.
3. Suy luận step-by-step:
   a) [bước suy luận 1 dựa trên dẫn chứng ở trên]
   b) [bước suy luận 2 tiếp theo]
   …
4. Kết luận: Đúng  
```



### 2. Model Training

With the augmented dataset of 728 samples, we proceeded with a two-phase training strategy.

#### Dataset Split

- **Total Samples:** 728
- **Training Set:** 628 samples
  - True/False: 332
  - Multiple-choice: 245
  - Free-text: 51
- **Validation Set:** 100 samples
  - True/False: 55
  - Multiple-choice: 40
  - Free-text: 5

#### Phase 1: Supervised Fine-Tuning (SFT)

We fine-tuned a model on the training set to teach it the Chain-of-Thought format.

- **Base Model:** `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`
- **Training Duration:** 14 epochs (approximately 12 hours).
- **Resulting Model:** `qwen_14e`

#### Phase 2: GPRO (Generalized Preference Ranking Optimization)

After SFT, we took the `qwen_14e` model and performed further training using GPRO. We experimented with **three different configurations**, each utilizing a reward function with different weights, to create three specialized GPRO models.

#### Model Training & Inference Prompt

The following prompt structure was used during training and inference. The model is required to generate its reasoning within the `<think>`...`</think>` tags before outputting the final, concise answer.

```  
Dựa vào bối cảnh bên dưới, hãy phân tích kỹ trước khi trả lời câu hỏi.

Loại câu hỏi: Đúng/Sai (format của Kết luận cuối cùng sau khi suy luận là 1 trong 2 kết luận: 'Đúng', 'Sai'. Không được giải thích gì thêm.)

Câu hỏi: Người nghiện ma túy từ đủ 18 tuổi trở lên bị áp dụng biện pháp xử lý hành chính đưa vào cơ sở cai nghiện bắt buộc theo quy định của Luật Xử lý vi phạm hành chính khi bị phát hiện sử dụng chất ma túy một cách trái phép trong thời gian cai nghiện ma túy tự nguyện, đúng hay sai?

Bối cảnh: 
Đối tượng bị áp dụng biện pháp xử lý hành chính đưa vào cơ sở cai nghiện bắt buộc

Người nghiện ma túy từ đủ 18 tuổi trở lên bị áp dụng biện pháp xử lý hành chính đưa vào cơ sở cai nghiện bắt buộc theo quy định của Luật Xử lý vi phạm hành chính khi thuộc một trong các trường hợp sau đây:

1. Không đăng ký, không thực hiện hoặc tự ý chấm dứt cai nghiện ma túy tự nguyện;

2. Trong thời gian cai nghiện ma túy tự nguyện bị phát hiện sử dụng trái phép chất ma túy;

3. Người nghiện ma túy các chất dạng thuốc phiện không đăng ký, không thực hiện hoặc tự ý chấm dứt điều trị nghiện các chất dạng thuốc phiện bằng thuốc thay thế hoặc bị chấm dứt điều trị nghiện các chất dạng thuốc phiện bằng thuốc thay thế do vi phạm quy định về điều trị nghiện;

4. Trong thời gian quản lý sau cai nghiện ma túy mà tái nghiện.

Hãy sinh phần suy luận chi tiết theo mẫu bên dưới trong thẻ <think>...</think>. Bạn cần viết đầy đủ các bước phân tích, dẫn chứng và suy luận trước khi đưa ra câu trả lời ngắn gọn.
Không được trả lời ngay mà phải suy luận đầy đủ trước trong <think>.

<think>
1. Phân tích câu hỏi: [trình bày ngắn gọn nội dung và ý định của câu hỏi]
2. Dẫn chứng từ bối cảnh:
   - Hãy tách từng đoạn dài trong bối cảnh thành nhiều ý nhỏ rõ ràng.
   - Hãy tách ít nhất 3 đến 5 ý ở phần dẫn chứng từ bối cảnh
   - Mỗi ý nên nêu rõ nội dung pháp lý, viết ngắn gọn dễ hiểu.
   - Ví dụ:
       - [ý 1 từ đoạn luật A]
       - [ý 2 từ đoạn luật A]
       - [ý 3 từ đoạn luật B]
   - Ghi rõ đoạn nào có liên quan đến câu hỏi.
3. Suy luận step-by-step:
   a) [bước suy luận 1 dựa trên dẫn chứng ở trên]
   b) [bước suy luận 2 tiếp theo]
   …
4. Kết luận: [tóm tắt câu trả lời cuối cùng dựa trên suy luận]
</think>

[Kết luận cuối cùng sau khi suy luận]  
```


### 3. Submission Strategies (Ensemble)

To maximize performance, we used ensemble methods combining the outputs of our trained models.

- **Submission 1:** Combined the answers from two SFT models: `qwen_3e` and `qwen_14e`.

- **Submission 2:** An ensemble of the **three GPRO-trained models** using a voting mechanism to determine the final answer.

- **Submission 3:** An ensemble combining the **three GPRO models and the `qwen_14e` SFT model** (total of 4 models), also using a voting mechanism.

### 4. How to Run

Follow these steps to reproduce the results.

- **Data Preparation (Generating the "think" component):**
  To augment the dataset with the reasoning part, run the notebook:
  `alqac-gen-data-train.ipynb`

- **SFT Training:**
  To perform Supervised Fine-Tuning (SFT) on the full dataset, run:
  `alqac-train-data-train-qwen-fulldata.ipynb`

- **GPRO Training:**
  To train the model using GPRO with the three different configurations, run these notebooks:
  - `alqac-train-grpo-config1.ipynb`
  - `alqac-train-grpo-config2.ipynb`
  - `alqac-train-grpo-config3.ipynb`

- **Inference:**
  - To perform inference on the test set using the SFT model, run:
    `alqac-inference-test-qwen-fulldata.ipynb`
  - To perform inference on the test set using the GPRO models, run:
    `alqac-inference-test-qwen-grpo-fulldata.ipynb`

- **Generate Submission Files:**
  To create the three final submission files by ensembling the model outputs, run:
  `submit.ipynb`