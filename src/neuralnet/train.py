import torch
from ignite.engine import Engine, Events
from ignite.metrics import ROC_AUC, Loss
from ignite.handlers import ModelCheckpoint
from ignite.contrib.handlers import global_step_from_engine, TensorboardLogger
from ignite.handlers.early_stopping import EarlyStopping
import matplotlib.pyplot as plt


def train_model(model, train_loader, val_loader, columns_idx, criterion, optimizer, epochs=10):
    def train_step(engine, batch):
        model.train()

        # Zero the parameter gradients
        optimizer.zero_grad()

        inputs, labels = batch[0], batch[1]
        input_ids = inputs[:, columns_idx["INPUT_IDS_START"] : columns_idx["INPUT_IDS_END"]]
        attention_mask = inputs[
            :, columns_idx["ATTENTION_MASK_START"] : columns_idx["ATTENTION_MASK_END"]
        ]
        x_cat = inputs[:, columns_idx["CATEGORICAL_START"] : columns_idx["CATEGORICAL_END"]]
        x_cont = inputs[:, columns_idx["NUMERIC_START"] : columns_idx["NUMERIC_END"]]

        # Forward pass
        outputs = model(input_ids, attention_mask, x_cat, x_cont).squeeze()
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        return loss.item()

    trainer = Engine(train_step)

    val_metrics = {"roc_auc_score": ROC_AUC(), "loss": Loss(criterion)}

    def validation_step(engine, batch):
        model.eval()
        with torch.no_grad():
            inputs, labels = batch[0], batch[1]
            input_ids = inputs[:, columns_idx["INPUT_IDS_START"] : columns_idx["INPUT_IDS_END"]]
            attention_mask = inputs[
                :, columns_idx["ATTENTION_MASK_START"] : columns_idx["ATTENTION_MASK_END"]
            ]
            x_cat = inputs[:, columns_idx["CATEGORICAL_START"] : columns_idx["CATEGORICAL_END"]]
            x_cont = inputs[:, columns_idx["NUMERIC_START"] : columns_idx["NUMERIC_END"]]

            # Forward pass
            outputs = model(input_ids, attention_mask, x_cat, x_cont).squeeze()
            return outputs, labels

    train_evaluator = Engine(validation_step)
    val_evaluator = Engine(validation_step)

    # Attach metrics to the evaluators
    for name, metric in val_metrics.items():
        metric.attach(train_evaluator, name)

    for name, metric in val_metrics.items():
        metric.attach(val_evaluator, name)

    log_interval = 100

    @trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
    def log_training_loss(engine):
        print(
            f"Epoch[{engine.state.epoch}], Iter[{engine.state.iteration}] Loss: {engine.state.output:.2f}"
        )

    @trainer.on(Events.EPOCH_COMPLETED(every=epochs // 25))
    def log_training_results(trainer):
        train_evaluator.run(train_loader)
        metrics = train_evaluator.state.metrics
        print(
            f"Training Results - Epoch[{trainer.state.epoch}] Avg roc_auc_score: {metrics['roc_auc_score']:.2f} Avg loss: {metrics['loss']:.2f}"
        )

    @trainer.on(Events.EPOCH_COMPLETED(every=epochs // 25))
    def log_validation_results(trainer):
        val_evaluator.run(val_loader)
        metrics = val_evaluator.state.metrics
        print(
            f"Validation Results - Epoch[{trainer.state.epoch}] Avg roc_auc_score: {metrics['roc_auc_score']:.2f} Avg loss: {metrics['loss']:.2f}"
        )

    def score_function(engine):
        return engine.state.metrics["roc_auc_score"]

    model_checkpoint = ModelCheckpoint(
        "checkpoint",
        n_saved=1,
        filename_prefix="best",
        score_function=score_function,
        score_name="roc_auc_score",
        global_step_transform=global_step_from_engine(trainer),
    )

    val_evaluator.add_event_handler(Events.COMPLETED, model_checkpoint, {"model": model})

    early_stopping = EarlyStopping(
        patience=10, score_function=lambda engine: -engine.state.metrics["loss"], trainer=trainer
    )
    # Note: the handler is attached to an *Evaluator* (runs one epoch on validation dataset).
    val_evaluator.add_event_handler(Events.COMPLETED, early_stopping)

    tb_logger = TensorboardLogger(log_dir="tb-logger")

    tb_logger.attach_output_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED(every=log_interval),
        tag="training",
        output_transform=lambda loss: {"batch_loss": loss},
    )

    for tag, evaluator in [("training", train_evaluator), ("validation", val_evaluator)]:
        tb_logger.attach_output_handler(
            evaluator,
            event_name=Events.EPOCH_COMPLETED,
            tag=tag,
            metric_names=["roc_auc_score", "loss"],
            global_step_transform=global_step_from_engine(trainer),
        )

    trainer.run(train_loader, max_epochs=epochs)

    tb_logger.close()
