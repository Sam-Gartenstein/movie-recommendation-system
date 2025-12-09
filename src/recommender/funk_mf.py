import numpy as np

class FunkMF:
    def __init__(self, n_factors=20, n_epochs=10, lr=0.01, reg=0.02, random_state=42):
        
        '''
        Model hyperparameters
        
        - Number of latent dimensions (k)
        - Number of full passes over the training data during SGD
        - Learning rate (gamma, which controls how the magnitidue of each SGD step
        - Regularization strength (lambda). Helps prevent the model from overfitting 
          by shrinking biases and latent factors
        - Random_state Seed for the random number generator
        '''
        self.n_factors = n_factors 
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.random_state = random_state
        
        '''
        Model parameters (learned during training):
        
        -self.mu (float):
            Global average rating across all users and items
        
        -self.b_u: np.dnarray of shape (num_users,)
            Vector of biases, where b_u[u] measures how much user u tends to 
            rate above or below the global mean
        
        - self.b_i : np.ndarray of shape (num_items,)
            Vector of (movie) biases, where b_i[i] measures how much item i 
            tends to be rated above or below the global mean.
        
        - self.P : np.ndarray of shape (num_users, n_factors)
            User latent factor matrix, where row P[u] is the latent feature vector
            p_u for user u (their preferences in the latent space)
        
        - self.Q : np.ndarray of shape (num_items, n_factors)
            Item latent factor matrix, where row Q[i] is the latent feature vector
            q_i for item i (its characteristics in the latent space).
        '''
        self.mu = None
        self.b_u = None
        self.b_i = None
        self.P = None
        self.Q = None
        
        '''
        ID mapping dictionaries (not learned parameters):
        
        - self.user_id_to_idx : dict
            Maps raw userId values from the data to internal user indices
            0..num_users-1.

        - self.movie_id_to_idx : dict
            Maps raw movieId values from the data to internal item indices
            0..num_items-1.

        - self.idx_to_user_id : dict
            Reverse mapping from internal user index back to raw userId.
            Useful when interpreting recommendations.

        - self.idx_to_movie_id : dict
            Reverse mapping from internal item index back to raw movieId.
        '''
        self.user_id_to_idx = None
        self.movie_id_to_idx = None
        self.idx_to_user_id = None
        self.idx_to_movie_id = None
        
        # Movie title mapping
        self.movie_id_to_title = None
        
    def fit(self, ratings_df):
        
        # Random number generator
        rng = np.random.default_rng(self.random_state)
        
        # ------------------------ Step 1: mappings ------------------------
        
        # Number of unique users and movies
        unique_users = ratings_df["userId"].unique()
        unique_movies = ratings_df["movieId"].unique()
        
        '''
        Build mappings between raw userId values and internal indices
        
        - self.user_id_to_idx: maps userId -> 0..num_users-1
         Used to locate the correct row in bu and P for a given user
              
        - self.idx_to_user_id: reverse mapping, internal index -> userId
         Used for converting model outputs (indices) back to real user IDs
        '''
        self.user_id_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
        self.idx_to_user_id = {idx: uid for uid, idx in self.user_id_to_idx.items()}    
        
        '''
        Build mappings between raw movieId values and internal item indices.

        - self.movie_id_to_idx: maps movieId -> 0..num_items-1
          Used to locate the correct row in b_i and Q for a given movie.

        - self.idx_to_movie_id: reverse mapping, internal index -> movieId
          Used for converting model outputs (indices) back to real movie IDs.
        '''        
        self.movie_id_to_idx = {mid: idx for idx, mid in enumerate(unique_movies)}
        self.idx_to_movie_id = {idx: mid for mid, idx in self.movie_id_to_idx.items()}
        
        # Total number of users and movies (items)
        num_users = len(self.user_id_to_idx)
        num_items = len(self.movie_id_to_idx)
        
        # ------------------------ Step 2: init params ------------------------
        
        # Global mean rating across all users and movies
        self.mu = ratings_df["rating"].mean()
        
        # Initialize user and item bias vectors to zero
        self.b_u = np.zeros(num_users)
        self.b_i = np.zeros(num_items)
        
        '''
        Initialize user (P) and item (Q) latent factor matrices with small
        random values. Each row of P is p_u (user u's latent preferences), 
        and each row of Q is q_i (item i's latent attributes). The factor 
        0.1 keeps initial values small to help stable training
        ''' 
        self.P = 0.1 * rng.standard_normal((num_users, self.n_factors))
        self.Q = 0.1 * rng.standard_normal((num_items, self.n_factors))
        
        # ------------------------ Step 3: build training triples ------------------------
        
        '''
        -Build a list of training triples (u_idx, i_idx, r_ui)
        -For each row in ratings_df, convert raw userId and movieId into internal
         indices using the mapping dicts, and keeping rating as-is
        '''
        train_data = [
            (self.user_id_to_idx[u], self.movie_id_to_idx[m], r)
            for u, m, r in ratings_df[["userId", "movieId", "rating"]].itertuples(index=False)
        ]
        
        # ------------------------ Step 4: SGD loop ------------------------
        
        # Perform self.n_epochs full passes (epochs) over the training data
        for epoch in range(self.n_epochs):
            
            '''
            Shuffle the (u_idx, i_idx, r_ui) triples so we don't always
            update parameters in the same order each epoch to prevent 
            overfitting when training the data
            '''
            rng.shuffle(train_data)
            
            '''
            Record and accumulate the sum of squared error over the current 
            epoch to monitor how well the the model is fitting
            '''
            total_sq_error = 0.0
            
            '''
            Loop over every observed rating (u_idx, i_idx, r_ui) and 
            perform one SGD update per triple
            '''
            for u_idx, i_idx, r_ui in train_data:
                total_sq_error += self._sgd_step(u_idx, i_idx, r_ui)

            '''
            Compute the mean squared error (MSE) for this epoch as an
            average over all training examples (print for training diagnostics)
            '''
            mse = total_sq_error / len(train_data)
            print(f"Epoch {epoch+1}/{self.n_epochs} - MSE: {mse:.4f}")

    def _predict_idx(self, u_idx, i_idx):
        '''
        Compute the predicted rating for a given user and item, working in index
        space (u_idx, i_idx).
        
        This implements the matrix factorization prediction formula:
            \hat{r}_{ui} = mu + b_u[u_idx] + b_i[i_idx] + P[u_idx] · Q[i_idx]
            
        where:
        - self.mu        : global mean rating
        - self.bu[u_idx] : bias term for user u
        - self.bi[i_idx] : bias term for item i
        - self.P[u_idx]  : latent factor vector p_u for user u
        - self.Q[i_idx]  : latent factor vector q_i for item i

        '''
        return (
            self.mu
            + self.b_u[u_idx]
            + self.b_i[i_idx]
            + self.P[u_idx].dot(self.Q[i_idx])
        )
    
    def _sgd_step(self, u_idx, i_idx, r_ui):
        """
        Perform one stochastic gradient descent (SGD) update for a single
        observed rating (user u_idx, item i_idx, rating r_ui).

        This updates:
        - user bias      self.bu[u_idx]
        - item bias      self.bi[i_idx]
        - user factors   self.P[u_idx]
        - item factors   self.Q[i_idx]

        according to the gradients of the regularized squared error loss.
        Returns the squared error for monitoring (e_ui**2).
        """
        
        # Compute current prediction \hat{r}_{ui} using the MF model and
        pred = self._predict_idx(u_idx, i_idx)
        
        # Residual error: e_ui = r_ui - \hat{r}_{ui} 
        e_ui = r_ui - pred

        
        #Update bias terms via gradient descent for b_u and b_i
        self.b_u[u_idx] += self.lr * (e_ui - self.reg * self.b_u[u_idx])
        self.b_i[i_idx] += self.lr * (e_ui - self.reg * self.b_i[i_idx])

        # ----- Update latent factor vectors -----
        
        '''
        Make copies of the current user/item factor vectors so that 
        when we update P[u] we still use the "old" Q[i], and vice versa
        '''
        Pu = self.P[u_idx].copy()
        Qi = self.Q[i_idx].copy()
        
        '''
        Update user and item factors:
        - e_ui * Qi moves the user vector P[u_idx] in a direction that
          reduces the prediction error for this item
        - e_ui * Pu does the same for the item vector Q[i_idx] with respect
          to this user
        - The - self.reg * Pu / Qi (not division) terms apply L2 regularization 
          so the factors do not grow too large (helps prevent overfitting)
        '''
        self.P[u_idx] += self.lr * (e_ui * Qi - self.reg * Pu)
        self.Q[i_idx] += self.lr * (e_ui * Pu - self.reg * Qi)

        '''
        Return squared error for this example so the caller
        can accumulate it and compute MSE over an epoch
        '''
        return e_ui**2
    
    def predict(self, user_id, movie_id, min_rating=None, max_rating=None):
        """
        Predict the rating that a given user would give to a given movie.

        Args:
            user_id: raw userId from the ratings data.
            movie_id: raw movieId from the movies data.
            min_rating: optional lower bound to clip predictions (e.g., 0.5 or 1.0).
            max_rating: optional upper bound to clip predictions (e.g., 5.0).

        Returns:
            A single predicted rating (float).
        """
        
        '''
        Look up the internal index for this user. If the user was seen
        during training, use their learned bias and latent factors. 
        If the user is new/unseen, fall back to zero bias and a zero vector
        for their latent factors.
        '''
        if user_id in self.user_id_to_idx:
            u_idx = self.user_id_to_idx[user_id]
            b_u = self.b_u[u_idx]
            Pu = self.P[u_idx]
        else:
            b_u = 0.0
            Pu = np.zeros(self.n_factors)

        '''
        Same logic for the movie: if we have seen this movie during training,
        use its learned bias and latent factors. Otherwise, use zeros.
        '''    
        if movie_id in self.movie_id_to_idx:
            i_idx = self.movie_id_to_idx[movie_id]
            b_i = self.b_i[i_idx]
            Qi = self.Q[i_idx]
        else:
            b_i = 0.0
            Qi = np.zeros(self.n_factors)
        
        '''
        Matrix Factorization:
            \hat{r}_{ui} = mu + b_u + b_i + p_u · q_i
        '''
        pred = self.mu + b_u + b_i + Pu.dot(Qi)

        '''
        Clip the prediction so it stays within the valid rating range (0–5).
        '''
        if min_rating is not None:
            pred = max(min_rating, pred)
        if max_rating is not None:
            pred = min(max_rating, pred)

        return pred
    
    def recommend_top_n(self, user_id, movies_df, ratings_df=None,
                        n=10, min_rating=0.0, max_rating=5.0):
        """
        Recommend the top-N movies for a given user.

        Args:
            user_id: raw userId.
            movies_df: DataFrame with at least columns ['movieId', 'title'].
            ratings_df: optional DataFrame with columns ['userId', 'movieId', 'rating'].
                        If provided, movies the user has already rated will be excluded
                        from the recommendations.
            n: number of movies to return.
            min_rating, max_rating: optional clipping bounds for predictions.

        Returns:
            List of (title, movieId, predicted_rating) tuples.
        """
        
        # Build mapping from movieId to title locally
        movie_id_to_title = dict(zip(movies_df["movieId"], movies_df["title"]))
        all_movie_ids = set(movies_df["movieId"].values)

        # Optionally filter out movies the user has already rated
        if ratings_df is not None and user_id in self.user_id_to_idx:
            seen_movie_ids = set(
                ratings_df.loc[ratings_df["userId"] == user_id, "movieId"].values
            )
            candidate_movie_ids = list(all_movie_ids - seen_movie_ids)
        else:
            candidate_movie_ids = list(all_movie_ids)

        # Predict for all candidate movies
        preds = [
            (mid, self.predict(user_id=user_id,
                               movie_id=mid,
                               min_rating=min_rating,
                               max_rating=max_rating))
            for mid in candidate_movie_ids
        ]

        # Sort by predicted rating (high to low) and keep top N
        preds_sorted = sorted(preds, key=lambda x: x[1], reverse=True)[:n]

        # Attach titles
        results = [
            (movie_id_to_title.get(mid, f"Movie {mid}"), mid, score)
            for mid, score in preds_sorted
        ]
        return results

