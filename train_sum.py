from models.indosum import IndoSum
from utils.preprocessor import Preprocessor

import lightning as L

if __name__ == '__main__':
    indosum_model = IndoSum(lr = 2e-5)
    indosum_pre = Preprocessor(
        max_length = 5,
        batch_size=100
    )
    
    indosum_pre.setup(stage = "fit")
    data = indosum_pre.train_dataloader()
    
    trainer = L.Trainer(
        accelerator = "gpu",
        max_epochs = 400,
    )
    
    trainer.fit(indosum_model, datamodule = indosum_pre)
    