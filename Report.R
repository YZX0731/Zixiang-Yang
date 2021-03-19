#read data
data <- read.csv('C:\\Users\\lenovo\\Downloads\\bank_personal_loan.csv')

#view the data
View(data)
skimr::skim(data)
hist(data$Age)
hist(data$Income)
hist(data$Education,breaks = c(0,1,2,3))
DataExplorer::plot_histogram(data[,-9], ncol = 3)
require('corrplot')
e<-cor(data,method='pearson')
corrplot(e,method = 'pie',type='lower')
require('GGally')
ggpairs(data_factor %>% select(Personal.Loan,Age,Experience,Income),
        aes(color=Personal.Loan))
require('dplyr')
require('mlr3')
require('mlr3verse')
require("mlr3viz")
data_factor<-transform(data,Personal.Loan=as.factor(Personal.Loan))
data2 <- data_factor %>% 
  select(-Experience,-ZIP.Code,-Online,-CreditCard)

#Split the data(used for logistic and trees)
task_bank <- TaskClassif$new(id='bank', backend=data2, target='Personal.Loan')
print(task_bank)
train_data <- sample(task_bank$nrow,0.8*task_bank$nrow)
traind <- data2[train_data,]
test_data <- setdiff(seq_len(task_bank$nrow),train_data)
testd <- data2[test_data,]

#logistic
learner <- lrn('classif.log_reg',predict_type = 'prob')
learner$train(task_bank,row_ids = train_data)
print(learner$model)

prediction <- learner$predict(task_bank,row_ids = test_data)
print(prediction)
measure = msr("classif.acc")
prediction$score(measure)
autoplot(prediction, type = "roc")

#tree
learner <- lrn('classif.rpart',predict_type = 'prob',cp=0.07185, minsplit=4)
train <- learner$train(task_bank,row_ids = train_data)
print(learner$model)
plot(learner$model,compress = TRUE,margin=0.1)
text(learner$model, use.n = TRUE, cex = 0.8)
prediction <- learner$predict(task_bank,row_ids = test_data)
print(prediction)
head(as.data.table(prediction))
measure <- msr("classif.acc")
prediction$score(measure)
require(precrec)
autoplot(prediction, type = "roc")
#optimization
library("paradox")
library("mlr3tuning")
search_space <- ps(
  cp = p_dbl(lower = 0.001, upper = 0.1),
  minsplit = p_int(lower = 1, upper = 10)
)
terminator <- trm("evals", n_evals = 10)
tuner <- tnr("random_search")

at <- AutoTuner$new(
  learner = learner,
  resampling = rsmp("holdout"),
  measure = msr("classif.ce"),
  search_space = search_space,
  terminator = terminator,
  tuner = tuner
)
grid <- benchmark_grid(
  task = task_bank,
  learner = list(at, lrn("classif.rpart")),
  resampling = rsmp("cv", folds = 3)
)
logger <- lgr::get_logger("bbotk")
logger$set_threshold("warn")
bmr <- benchmark(grid)
bmr$aggregate(msrs(c("classif.ce", "classif.acc",'classif.fpr','classif.fnr')))
t <- at$train(task_bank,row_ids = train_data)
p <- at$predict(task_bank,row_ids = test_data)
measure <- msr("classif.acc")
p$score(measure)
autoplot(prediction, type = "roc")
autoplot(p,type = 'roc')
print(at$model)
plot(at$model,compress = TRUE,margin=0.1)
text(at$model, use.n = TRUE, cex = 0.8)
#c50 boosting(***)
library(C50)
c50.model<-C5.0(traind,traind$Personal.Loan,trials=5)
summary(c50.model)
test<-predict(c50.model,testd)
confusion=table(testd$Personal.Loan,test)
confusion
accuracy=sum(diag(confusion))*100/sum(confusion)
accuracy

#neural network
data3 <- data %>% 
  select(-Experience,-ZIP.Code,-Online,-CreditCard)
train_row <- sample(1:nrow(data3),dim(data3)[1]*0.6)
traindata<-data3[train_row,]
remain<-data3[-train_row,]
valid_row<-sample(1:nrow(remain),dim(data3)[1]*0.2)
validdata<-remain[valid_row,]
testdata<-remain[-valid_row,]
require('nnet')
nnet.model <- nnet(Personal.Loan~.,traindata,size=10,decay=0.01)
summary(nnet.model)
require('keras')
bank_train_x <- traindata[,-7] %>%
  as.matrix()
bank_train_y <- traindata[,7] %>%
  as.matrix()

bank_validate_x <- validdata[,-7] %>%
  as.matrix()
bank_validate_y <- validdata[,7] %>%
  as.matrix()

bank_test_x <- testdata[,-7] %>%
  as.matrix()
bank_test_y <- testdata[,7] %>%
  as.matrix()
deep.net <- keras_model_sequential() %>%
  layer_dense(units = 32, activation = "relu",
              input_shape = c(ncol(bank_train_x))) %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 1, activation = "sigmoid")
deep.net %>% compile(
  loss = "binary_crossentropy",
  optimizer = 'adam',
  metrics = c("accuracy")
)
deep.net
deep.net %>% fit(
  bank_train_x, bank_train_y,
  epochs = 100, batch_size = 128,
  validation_data = list(bank_validate_x, bank_validate_y),
)
pred_test_prob <- deep.net %>% predict_proba(bank_test_x)
pred_test_res <- deep.net %>% predict_classes(bank_test_x)
table(pred_test_res, bank_test_y)
yardstick::accuracy_vec(as.factor(bank_test_y),
                        as.factor(pred_test_res))
yardstick::roc_auc_vec(factor(bank_test_y, levels = c("1","0")),
                       c(pred_test_prob))
