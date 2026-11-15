from flcore.clients.clientfedload import clientFedLoad
import torch


class clientHERAFL(clientFedLoad):
    """HERAFL uses the same submodel training; its policy runs on the server."""

    def training_class_ids(self):
        if not hasattr(self, '_training_class_ids'):
            # Use the actual local training set, including few-shot filtering.
            self._training_class_ids = sorted({int(y) for _, y in self.load_train_data().dataset})
        return self._training_class_ids

    def local_accuracy_metrics(self):
        """Measure the trained local submodel before global aggregation.

        Absent means absent from this UE's training data. The downloaded global
        weights may already contain knowledge from other clients.
        """
        seen_ids = torch.tensor(self.training_class_ids(), device=self.device)
        seen_correct = seen_samples = absent_correct = absent_samples = 0
        self.model.eval()
        with torch.no_grad():
            for x, y in self.load_test_data():
                x, y = x.to(self.device), y.to(self.device)
                correct = self.model(x).argmax(1) == y
                seen = torch.isin(y, seen_ids)
                seen_samples += int(seen.sum())
                absent_samples += int((~seen).sum())
                seen_correct += int((correct & seen).sum())
                absent_correct += int((correct & ~seen).sum())
        self.round_stats.update(
            local_seen_correct=seen_correct, local_seen_samples=seen_samples,
            local_seen_accuracy=seen_correct / seen_samples if seen_samples else None,
            local_absent_correct=absent_correct, local_absent_samples=absent_samples,
            local_absent_accuracy=absent_correct / absent_samples if absent_samples else None,
            local_train_class_count=len(seen_ids))
        # Preserve the existing reward: accuracy over all local test samples.
        return seen_correct + absent_correct, seen_samples + absent_samples
