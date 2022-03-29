# -*- coding: utf-8 -*-
import platform
from io import BytesIO, StringIO
from unittest.main import MODULE_EXAMPLES
import matplotlib.pyplot as plt
import matplotlib
import base64

import json
import numpy as np
import tornado.web
import pandas as pd
from sklearn import metrics
from sklearn import linear_model
from sklearn import ensemble
from sklearn import model_selection

CLASSIFIER_SCORES = ["accuracy", "f1", "roc_auc", "precision", "recall"]
REGRESSION_SCORES = ["neg_mean_absolute_error", "neg_root_mean_squared_error"]

FORM_FEATURES = ["model_class", "sheet_name", "feature_columns", "label_column", "train_ratio", "model_type", "cross_validation", "model_params_v"]

MODEL_MAP = {
    "classifier_logistic": linear_model.LogisticRegression,
    "classifier_gbdt": ensemble.GradientBoostingClassifier,
    "regression_linear": linear_model.LinearRegression,
    "regression_gbdt": ensemble.GradientBoostingRegressor
}

class ModelHandler(tornado.web.RequestHandler):
    """
    model handler
    """

    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "x-requested-with")
        self.set_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')

    def get(self):
        self.post()

    def data_format(self, http_files, params_dict):
        sheet_name = params_dict["sheet_name"]
        file_lst = http_files.get("raw_data_file", None)
        if not file_lst or sheet_name is None:
            return None
        http_file = file_lst[0] 

        df = None
        if http_file.filename.endswith(".xlsx"):
            if platform.system() == "Darwin":
                df = pd.read_excel(http_file.body, engine='openpyxl', sheet_name=sheet_name)
            else:
                df = pd.read_excel(http_file.body, sheet_name=sheet_name)
        elif http_file.filename.endswith(".csv") or http_file.filename.endswith(".txt"):
            s = str(http_file.body,'utf-8')
            data = StringIO(s) 
            df = pd.read_csv(data)
        else:
            pass

        return df

    def post(self):

        raw_data_file = self.request.files
        params_dict = dict()
        result_dict = {
            "msg": "Success",
            "total_sample_cnt": 0,
            "train_sample_cnt": None,
            "test_sample_cnt": None,
            "img_data": None
        }

        for feature_name in FORM_FEATURES:
            v = self.get_arguments(feature_name)
            
            if feature_name == "feature_columns":
                params_dict[feature_name] = v[0].strip().split(",") if v else None
            elif feature_name == "train_ratio":
                params_dict[feature_name] = float(v[0].strip()) if v and v[0] else None
            elif feature_name == "model_params_v":
                params_dict[feature_name] = json.loads(v[0] if v and v[0] else "{}")
            elif feature_name == "cross_validation":
                params_dict[feature_name] = int(v[0].strip()) if v and v[0] else None
            else:
                params_dict[feature_name] = v[0] if v else None

        df = self.data_format(raw_data_file, params_dict)
        model_raw = MODEL_MAP.get("_".join([params_dict["model_class"], params_dict["model_type"]]), None)
        
        score_items = CLASSIFIER_SCORES if params_dict["model_class"] == "classifier" else REGRESSION_SCORES

        if df is None or model_raw is None:
            if df is None:
                result_dict["msg"] = "read data error"
            elif model_raw is None:
                result_dict["msg"] = "invalid model_class params"
            self.finish(result_dict)
            return

        result_dict["total_sample_cnt"] = len(df) 
        
        if params_dict["cross_validation"] not in {"", None}:
            model = model_raw(**params_dict["model_params_v"])
            scores = model_selection.cross_validate(
                model, 
                df[params_dict["feature_columns"]], 
                df[params_dict["label_column"]], 
                cv=params_dict["cross_validation"], 
                scoring=score_items
            )
            
            score_results = {
                k: np.mean(scores[f"test_{k}"]) for k in score_items
            }

            result_dict.update(score_results)
        else:
            model = model_raw(**params_dict["model_params_v"])
            x_train, x_test, y_train, y_test = model_selection.train_test_split(
                df[params_dict["feature_columns"]], 
                df[params_dict["label_column"]], 
                train_size=params_dict["train_ratio"]
            )
            result_dict["train_sample_cnt"] = len(x_train)
            result_dict["test_sample_cnt"] = len(x_test)

            # 训练
            model.fit(x_train[params_dict["feature_columns"]], y_train)

            # 预测0/1
            y_predict = model.predict(x_test[params_dict["feature_columns"]])

            if params_dict["model_class"] == "classifier":
                # 预测概率
                y_predict_proba = model.predict_proba(x_test[params_dict["feature_columns"]])
                y_predict_proba = y_predict_proba[:, 1]

                # 计算 分值
                result_dict["accuracy"] = metrics.accuracy_score(y_test, y_predict)
                result_dict["roc_auc"] = metrics.roc_auc_score(y_test, y_predict_proba)
                result_dict["f1"] = metrics.f1_score(y_test, y_predict)
                result_dict["precision"] = metrics.precision_score(y_test, y_predict)
                result_dict["recall"] = metrics.recall_score(y_test, y_predict)
                
                # ROC曲线
                fpr, tpr, threshold = metrics.roc_curve(y_test, y_predict_proba)
                roc_auc = metrics.auc(fpr, tpr)
                fig = plt.figure(figsize=(4.2,4))
                matplotlib.rcParams.update({'font.size': 10})
                plt.title('ROC curve for testing set')
                plt.plot(fpr, tpr, 'b', label = 'Val AUC = %0.3f' % roc_auc)
                plt.legend(loc = 'lower right')
                plt.plot([0, 1], [0, 1],'r--')
                plt.xlim([0, 1])
                plt.ylim([0, 1])
                plt.ylabel('True Positive Rate (Sensitivity)')
                plt.xlabel('False Positive Rate (1 - Specificity)')
                canvas = fig.canvas
                buffer = BytesIO()
                canvas.print_png(buffer)
                result_dict["img_data"] = (b"data:image/png;base64," + base64.b64encode(buffer.getvalue())).decode("utf-8")
                buffer.close()
                
        self.finish(result_dict)