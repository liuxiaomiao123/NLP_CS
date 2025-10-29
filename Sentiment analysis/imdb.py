import numpy as np
from pathlib import Path
from argparse import ArgumentParser, Namespace, BooleanOptionalAction
from transformers import set_seed
from torch import device as tdevice, no_grad, save as torch_save
from torch.optim import Adam
from torch.nn import BCELoss
from tqdm import tqdm
from data_processing import *
from lstm import MovieLSTM
from sklearn.metrics import f1_score




def create_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--fname', type=Path, default='IMDB Dataset.csv')
    parser.add_argument('--train_prop', type=float, default=0.8)
    parser.add_argument('--test_prop', type=float, default=0.1)
    parser.add_argument('--val_prop', type=float, default=0.1)
    parser.add_argument('--padding_len', type=int, default=256)
    parser.add_argument('--max_epochs', type=int, default=3)
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--device_str', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=24)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--save_best_model', action=BooleanOptionalAction, default=True)
    parser.add_argument('--shuffle', action=BooleanOptionalAction, default=False)
    parser.add_argument('--num_most_freq_words', type=int, default=10000)
    parser.add_argument('--num_hidden_layers', type=int, default=2)
    parser.add_argument('--dim_embed', type=int, default=100)
    parser.add_argument('--dim_hidden', type=int, default=256)
    parser.add_argument('--dropout_perc', type=float, default=0.3)
    parser.add_argument('--batch_first', action=BooleanOptionalAction, default=True)
    args = parser.parse_args()
    return args

def evaluate(model, loader, device, batch_size):
    model.eval()
    all_preds = []
    all_labels = []
    with no_grad():
        hidden_states = model.initialize_hidden_states(b_size=batch_size)
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            hidden_states = tuple([state.data for state in hidden_states])
            lstm_out, hidden_states = model(X, hidden_states)

            preds = (lstm_out > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    return f1_score(all_labels, all_preds)


def validation(model, dev_loader, device, batch_size):
    f1_validation = evaluate(model, dev_loader, device=device, batch_size=batch_size)
    print(f'Validation F1 Score: {f1_validation:.4f}')
    return f1_validation

def test(model, test_loader, device, batch_size):
    f1_test = evaluate(model, test_loader, device=device, batch_size=batch_size)
    print(f'Test F1 Score: {f1_test:.4f}')
    return f1_test


def main():
    args = create_args()

    set_seed(args.seed)
    np.random.seed(args.seed)

    stop_words = set(stopwords.words('english'))

    df = read_data(fname=args.fname)
    cleaned_df = clean_data(review_frame=df, stop_words=stop_words)

    vocab2idx = get_word_count(review_frame=cleaned_df)

    tokenizer = MovieTokenizer(vocab2idx=vocab2idx, padding_len=args.padding_len)
    tokenized_data = tokenizer.tokenize_inputs(review_frame=cleaned_df)
    padded_data = tokenizer.pad_reviews(tokenized_data=tokenized_data)
    labels = df['sentiment'].apply(lambda sentiment: 0 if sentiment == 'negative' else 1).to_list()
    train_dl, dev_dl, test_dl = train_test_split(
        padded_tokens=padded_data, labels=labels, seed=args.seed,
        batch_size=args.batch_size, train_prop=args.train_prop, 
        val_prop=args.val_prop, test_prop=args.test_prop, shuffle=args.shuffle
    )


    device = tdevice(args.device_str)

    model = MovieLSTM(
        vocab_size=len(vocab2idx)+1, dim_hidden=args.dim_hidden,
        dim_embed=args.dim_embed, num_hidden_layers=args.num_hidden_layers,
        dropout_perc=args.dropout_perc, batch_first=args.batch_first, device=device
    )
    model.to(device)

    loss_fnc = BCELoss()
    optim = Adam(model.parameters(), lr=args.lr)

    for epoch in tqdm(range(args.max_epochs)):
        model.train()
        hidden_states = model.initialize_hidden_states(b_size=args.batch_size)
        all_preds = []
        all_labels = []
        for i, (X, Y) in tqdm(enumerate(train_dl)):
            optim.zero_grad()
            X, Y = X.to(device), Y.to(device).float()
            hidden_states = tuple([state.data for state in hidden_states])
            lstm_out, hidden_states = model(X, hidden_states)
            loss = loss_fnc(lstm_out.squeeze(), Y)
            loss.backward()
            optim.step()
            preds = (lstm_out > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(Y.cpu().numpy())
        train_score = f1_score(all_labels, all_preds)
        print(f'Train F1 Score: {train_score:.4f}')
        validation(model=model, dev_loader=dev_dl, device=device, batch_size=args.batch_size)
    test(model=model, test_loader=test_dl, device=device, batch_size=args.batch_size)      

    if args.save_best_model:
        torch_save(model.state_dict(), args.output_dir + r'\best_model.pt')    



if __name__ == '__main__':
    main()