import torch
import torch.nn as nn
import torch.optim as optim
import time

# from dataset_wiki import vocab, max_seq_len, train_dataloader, valid_dataloader
from dataset_wiki_hf import vocab, max_seq_len, train_dataloader, valid_dataloader
from model_gpt import GPT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

model = GPT(len(vocab), max_seq_len).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=vocab["<pad>"])
optimizer = optim.Adam(model.parameters(), lr=0.0001)
epochs = 2

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"模型总参数量: {total_params:,}")
print(f"可训练参数量: {trainable_params:,}")
print(f"词汇表大小: {len(vocab)}")
print(f"最大序列长度: {max_seq_len}")
print(f"训练批次数: {len(train_dataloader)}")
print(f"验证批次数: {len(valid_dataloader)}")
print("-" * 50)

min_valid_loss = float("inf")
save_path = "best_model.pth"

for epoch in range(epochs):
    epoch_start_time = time.time()
    model.train()
    epoch_loss = 0
    batch_losses = []
    
    print(f"\nEpoch {epoch + 1}/{epochs}")
    print("-" * 30)
    
    for batch_idx, (source, target) in enumerate(train_dataloader):
        batch_start_time = time.time()
        
        inputs, targets = source.to(device), target.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.view(-1, len(vocab)), targets.view(-1))
        loss.backward()
        
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        batch_loss = loss.item()
        epoch_loss += batch_loss
        batch_losses.append(batch_loss)
        batch_time = time.time() - batch_start_time
        
        if (batch_idx + 1) % 100 == 0:
            progress = (batch_idx + 1) / len(train_dataloader) * 100
            avg_loss_so_far = epoch_loss / (batch_idx + 1)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Batch {batch_idx + 1:4d}/{len(train_dataloader)} "
                  f"({progress:5.1f}%) | "
                  f"Loss: {batch_loss:.4f} | "
                  f"Avg: {avg_loss_so_far:.4f} | "
                  f"Grad: {grad_norm:.4f} | "
                  f"Time: {batch_time:.2f}s")
    
    epoch_time = time.time() - epoch_start_time
    avg_train_loss = epoch_loss / len(train_dataloader)
    min_loss = min(batch_losses)
    max_loss = max(batch_losses)
    
    print(f"\n训练统计:")
    print(f"  平均损失: {avg_train_loss:.4f}")
    print(f"  最小损失: {min_loss:.4f}")
    print(f"  最大损失: {max_loss:.4f}")
    print(f"  Epoch耗时: {epoch_time:.2f}s")

    model.eval()
    valid_loss = 0
    valid_losses = []
    valid_start_time = time.time()
    
    with torch.no_grad():
        for source, target in valid_dataloader:
            inputs, targets = source.to(device), target.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, len(vocab)), targets.view(-1))
            valid_loss += loss.item()
            valid_losses.append(loss.item())
    
    valid_time = time.time() - valid_start_time
    avg_valid_loss = valid_loss / len(valid_dataloader)
    
    print(f"\n验证统计:")
    print(f"  平均损失: {avg_valid_loss:.4f}")
    print(f"  最小损失: {min(valid_losses):.4f}")
    print(f"  最大损失: {max(valid_losses):.4f}")
    print(f"  验证耗时: {valid_time:.2f}s")

    if avg_valid_loss < min_valid_loss:
        min_valid_loss = avg_valid_loss
        torch.save(model.state_dict(), save_path)
        print(f"\n✓ 新最佳模型已保存! 验证损失: {avg_valid_loss:.4f}")
    else:
        print(f"\n✗ 未改善 (最佳: {min_valid_loss:.4f})")
    
    print("=" * 50)


def generate_text_beam_search(model, input_str, max_len=50, beam_width=5):
    model.eval()
    input_tokens = [vocab[token] if token in vocab else vocab["<pad>"] for token in input_str.split()]
    candidates = [(input_tokens, 0.0)]
    
    with torch.no_grad():
        for step in range(max_len):
            new_candidates = []
            for candidate, candidate_score in candidates:
                inputs = torch.LongTensor(candidate).unsqueeze(0).to(device)
                outputs = model(inputs)
                logits = outputs[:, -1, :]
                scores, next_tokens = torch.topk(logits, beam_width, dim=-1)
                for score, next_token in zip(scores.squeeze(), next_tokens.squeeze()):
                    new_candidate = candidate + [next_token.item()]
                    new_score = candidate_score - score.item()
                    if next_token.item() != vocab["<eos>"]:
                        new_candidates.append((new_candidate, new_score))
            candidates = sorted(new_candidates, key=lambda x: x[1])[:beam_width]
            if not candidates:
                break
    
    if not candidates:
        return input_str
    
    best_candidate, _ = sorted(candidates, key=lambda x: x[1])[0]
    output_str = " ".join([vocab.get_itos()[token] for token in best_candidate if vocab.get_itos()[token] not in ["<pad>", "<sos>"]])
    return output_str

print("\n" + "=" * 50)
print("加载最佳模型并生成文本...")
print("=" * 50)

model.load_state_dict(torch.load(save_path, weights_only=True))
input_str = "my name"
generated_text = generate_text_beam_search(model, input_str)
print(f"\n输入: {input_str}")
print(f"生成: {generated_text}")
