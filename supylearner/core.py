from sklearn import clone, metrics
from sklearn.base import BaseEstimator, RegressorMixin
import sklearn.cross_validation as cv
import numpy as np
from scipy.optimize import fmin_l_bfgs_b, nnls

class SLError(Exception):
    """
    Base class for errors in the SupyLearner package
    """
    pass


class SuperLearner(BaseEstimator):
    """
    Loss-based super learning

    SuperLearner chooses a weighted combination of candidate estimates
    in a specified library using cross-validation.

    Parameters
    ----------
    library : list
        List of scikit-learn style estimators with fit() and predict()
        methods.

    K : Number of folds for cross-validation.

    loss : loss function, 'L2' or 'nloglik'.

    discrete : True to choose the best estimator
               from library ("discrete SuperLearner"), False to choose best
               weighted combination of esitmators in the library.

    coef_method : Method for estimating weights for weighted combination
                  of estimators in the library. 'L_BFGS_B' or 'NNLS'.

    Attributes
    ----------

    n_estimators : number of candidate estimators in the library.

    coef : Coefficients corresponding to the best weighted combination
           of candidate estimators in the libarary. 

    risk_cv : List of cross-validated risk estimates for each candidate
              estimator, and the (not cross-validated) estimated risk for
              the SuperLearner

    Examples
    --------

    """
    
    def __init__(self, library, K=5, loss='L2', discrete=False, coef_method='L_BFGS_B',\
                 save_pred_cv=False, bound=0.00001):
        self.library=library[:]
        self.K=K
        self.loss=loss
        self.discrete=discrete
        self.coef_method=coef_method
        self.n_estimators=len(library)
        self.save_pred_cv=save_pred_cv
        self.bound=bound
    
    def fit(self, X, y):
        """
        Fit SuperLearner.

        Parameters
        ----------
        X : numpy array of shape [n_samples,n_features]
            or other object acceptable to the fit() methods
            of all candidates in the library        
            Training data
        y : numpy array of shape [n_samples]
            Target values
        Returns
        -------
        self : returns an instance of self.
        """
        
        n=len(y)
        folds = cv.KFold(n, self.K)

        y_pred_cv = np.empty(shape=(n, self.n_estimators))
        for train_index, test_index in folds:
            X_train, X_test=X[train_index], X[test_index]
            y_train, y_test=y[train_index], y[test_index]
            for aa in range(self.n_estimators):
                est=clone(self.library[aa])
                est.fit(X_train,y_train)
        
                y_pred_cv[test_index, aa]=self._get_pred(est, X_test)
    
        self.coef=self._get_coefs(y, y_pred_cv)

        self.fitted_library=clone(self.library)
        for est in self.fitted_library:
            est.fit(X, y)
            
        self.risk_cv=[]
        for aa in range(self.n_estimators):
            self.risk_cv.append(self._get_risk(y, y_pred_cv[:,aa]))
        self.risk_cv.append(self._get_risk(y, self._get_combination(y_pred_cv, self.coef)))

        if self.save_pred_cv:
            self.y_pred_cv=y_pred_cv

        return self
                        
    
    def predict(self, X):
        """
        Predict using SuperLearner

        Parameters
        ----------
        X : numpy.array of shape [n_samples, n_features]
           or other object acceptable to the predict() methods
           of all candidates in the library
          


        Returns
        -------
        array, shape = [n_samples]
           Array containing the predicted class labels.
        """
        
        n_X = X.shape[0]
        y_pred_all = np.empty((n_X,self.n_estimators))
        for aa in range(self.n_estimators):
            y_pred_all[:,aa]=self._get_pred(self.fitted_library[aa], X)
        y_pred=self._get_combination(y_pred_all, self.coef)
        return y_pred


    def summarize(self):
        """
        Print CV risk estimates for each candidate estimator in the library,
        coefficients for weighted combination of estimators,
        and estimated risk for the SuperLearner.
        """
        
        print "Cross-validated risk estimates for each estimator in the library:"
        print self.risk_cv[:-1]
        print "Coefficients:", self.coef
        print "Estimated risk for SL:", self.risk_cv[-1]

    def _get_combination(self, y_pred_mat, coef):
        """
        Calculate weighted combination of predictions
        """
        if self.loss=='L2':
            comb=np.dot(y_pred_mat, coef)
        elif self.loss=='nloglik':
            comb=_inv_logit(np.dot(_logit(_trim(y_pred_mat, self.bound)), coef))
        return comb

    def _get_risk(self, y, y_pred):
        if self.loss=='L2':
            risk=np.mean((y-y_pred)**2)
        elif self.loss=='nloglik':
            risk=-np.mean( y   *   np.log(_trim(y_pred, self.bound))+\
                         (1-y)*np.log(1-(_trim(y_pred, self.bound))) )
        return risk
        
        
        
    
    def _get_coefs(self, y, y_pred_cv):
        if self.coef_method is 'L_BFGS_B':
            def ff(x):
                return self._get_risk(y, self._get_combination(y_pred_cv, x))
            x0=np.array([1./self.n_estimators]*self.n_estimators)
            bds=[(0,1)]*self.n_estimators
            a,b,c=fmin_l_bfgs_b(ff, x0, bounds=bds, approx_grad=True)
            coef=a/sum(a)
        elif self.coef_method is 'NNLS':
            if self.loss=='nloglik':
                raise SLError("coef_method 'NNLS' is only for 'L2' loss")
            init_coef, rnorm=nnls(y_pred_cv, y)
            coef=init_coef/sum(init_coef)        
        else: raise ValueError("method not recognized")
        return coef

    def _get_pred(self, est, X):
        """
        Get prediction from the estimator.
        Use est.predict if loss is L2.
        If loss is nloglik, use est.predict_proba if possible
        otherwise just est.predict, which hopefully returns something
        like a predicted probability, and not a class prediction.
        """
        if self.loss == 'L2':
            pred=est.predict(X)
        if self.loss == 'nloglik':
            if hasattr(est, "predict_proba"):
                #There should be a better way to do this
                #for SVM classifier
                if est.__class__.__name__ == "SVC":
                    pred=est.predict_proba(X)[:, 0]

                #for logistic regression
                elif est.__class__.__name__ == "LogisticRegression":
                    pred=est.predict_proba(X)[:, 1]
                else:
                    pred=est.predict_proba(X)
            else:
                pred=est.predict(X)
                if pred.min() < 0 or pred.max() > 1:
                    raise SLError("Probability less than zero or greater than one")
        return pred

def _trim(p, bound):
    p[p<bound]=bound
    p[p>1-bound]=1-bound
    return p

def _logit(p):
    return np.log(p/(1-p))

def _inv_logit(x):
    return 1/(1+np.exp(-x))
    
    
        
    

def cv_superlearner(library, X, y, K):
    sl=SuperLearner(library)
    library=library+[sl]

    
    n=len(y)
    folds=cv.KFold(n, K)
    y_pred_cv = np.empty(shape=(n, len(library)))

    for train_index, test_index in folds:
        X_train, X_test=X[train_index], X[test_index]
        y_train, y_test=y[train_index], y[test_index] 
        for aa in range(len(library)):
            est=library[aa]
            est.fit(X_train,y_train)
            y_pred_cv[test_index, aa]=_get_pred(est, X_test)

    risk_cv=[]        
    for aa in range(len(library)):
        risk_cv.append(metrics.mean_square_error(y, y_pred_cv[:,aa]))
    
    print risk_cv
    

    
