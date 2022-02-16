#from engine import evaluate
from DataPipeline import *
from Model import *

def training():
  criterion = DiceLoss()
  accuracy_metric = IoU()
  num_epochs=180
  valid_loss_min = np.Inf

  checkpoint_path = 'model_weight/chkpoint_'
  best_model_path = 'model_weight/bestmodel.pt'

  total_train_loss = []
  total_train_score = []
  total_valid_loss = []
  total_valid_score = []

  losses_value = 0
  for epoch in range(num_epochs):
    
      train_loss = []
      train_score = []
      valid_loss = []
      valid_score = []
      #<-----------Training Loop---------------------------->
      pbar = tqdm(train_loader, desc = 'description')
      for x_train, y_train in pbar:
        x_train = torch.autograd.Variable(x_train).cuda()
        y_train = torch.autograd.Variable(y_train).cuda()
        optimizer.zero_grad()
        output = model(x_train)
        #Loss
        loss = criterion(output, y_train)
        losses_value = loss.item()
        #Score
        score = accuracy_metric(output,y_train)
        loss.backward()
        optimizer.step()
        train_loss.append(losses_value)
        train_score.append(score.item())
        #train_score.append(score)
        pbar.set_description(f"Epoch: {epoch+1}, loss: {losses_value}, IoU: {score}")

      #<---------------Validation Loop---------------------->
      with torch.no_grad():
        for image,mask in val_loader:
          image = torch.autograd.Variable(image).cuda()
          mask = torch.autograd.Variable(mask).cuda()
          output = model(image)
          ## Compute Loss Value.
          loss = criterion(output, mask)
          losses_value = loss.item()
          ## Compute Accuracy Score
          score = accuracy_metric(output,mask)
          valid_loss.append(losses_value)
          valid_score.append(score.item())

      total_train_loss.append(np.mean(train_loss))
      total_train_score.append(np.mean(train_score))
      total_valid_loss.append(np.mean(valid_loss))
      total_valid_score.append(np.mean(valid_score))
      print(f"\n###############Train Loss: {total_train_loss[-1]}, Train IOU: {total_train_score[-1]}###############")
      print(f"###############Valid Loss: {total_valid_loss[-1]}, Valid IOU: {total_valid_score[-1]}###############")

      #Save best model Checkpoint
      # create checkpoint variable and add important data
      checkpoint = {
          'epoch': epoch + 1,
          'valid_loss_min': total_valid_loss[-1],
          'state_dict': model.state_dict(),
          'optimizer': optimizer.state_dict(),
      }
      
      # save checkpoint
      save_ckp(checkpoint, False, checkpoint_path, best_model_path)
      
      ## TODO: save the model if validation loss has decreased
      if total_valid_loss[-1] <= valid_loss_min:
          print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,total_valid_loss[-1]))
          # save checkpoint as best model
          save_ckp(checkpoint, True, checkpoint_path, best_model_path)
          valid_loss_min = total_valid_loss[-1]