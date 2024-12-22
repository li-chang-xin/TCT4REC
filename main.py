import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel
import warnings
from utils import *
from train import *
from ML_dataloader import *
from model.tctrec import TCTRec
from model.sasrec import SASRec
from model.gru4rec import GRU4Rec
from model.bsarec import BSARec

MODEL_DICT = {
    'tctrec': TCTRec,
    'sasrec' : SASRec,
    'gru4rec': GRU4Rec,
    'bsarec': BSARec
}


def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    args = parse_args()
    log_path = os.path.join(args.output_dir, args.data_name, args.model_type + '.log')
    logger = set_logger(log_path)
    
    check_path(args.output_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    length = args.max_seq_length
    args.checkpoint_path = os.path.join(args.output_dir, args.data_name ,args.model_type + '.pt')
    print('---------------start-----------------')


    data = get_dataset(args, logger)
    train_dataloader, valid_dataloader, test_dataloader = get_dataloader(args, data)

    logger.info(str(args))

    model = MODEL_DICT[args.model_type.lower()](args=args)
    logger.info(model)

    trainer = Train(model, data, train_dataloader, valid_dataloader, test_dataloader, args, logger)
    if args.do_eval:
        set_seed(args.seed)
        if args.load_model is None:
            logger.info(f"No model input!")
            exit(0)
        else:
            args.checkpoint_path = os.path.join(args.output_dir, args.data_name, args.load_model + '.pt')
            trainer.load(args.checkpoint_path)
            logger.info(f"Load model from {args.checkpoint_path} for test!")
            trainer.evaluate(epoch=1, mode='test')

    else:
        early_stopping = EarlyStopping(args.checkpoint_path, logger=logger, patience=args.patience, verbose=True)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        
        logger.info("training start")
        for epoch in range(args.num_epochs):
            #trainer.evaluate(epoch, mode='validate')####################
            trainer.train(epoch, optimizer, scheduler)
            result = list(trainer.evaluate(epoch, mode='validate'))
            early_stopping(result, trainer.model)
            if early_stopping.early_stop:
                logger.info("Early stopping")
                break

        logger.info("---------------Test Score---------------")
        trainer.model.load_state_dict(torch.load(args.checkpoint_path))
        trainer.evaluate(epoch=0, mode='test')

    logger.info(args.train_name)

if __name__ == '__main__':
    main()
