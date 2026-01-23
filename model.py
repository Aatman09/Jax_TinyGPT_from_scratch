import jax.numpy as jnp 
import jax


# for testing
key = jax.random.PRNGKey(0)

class RMSnorm:
    def __init__(self, * ,dim , epsilon = 1e-6):
        self.dim = dim
        self.epsilon = epsilon

        #Traniable parameter
        self.weights = jnp.ones((dim,))
    
    def __call__(self, x):
        rms = jnp.mean(x**2 , axis = -1 , keepdims=True)
        rms = jnp.sqrt(rms + self.epsilon)
        normalized_x = x / rms

        return normalized_x * self.weights


class Linear:
    def __init__(self ,* , in_features , out_features , key ):
        
        self.in_features = in_features
        self.out_features = out_features 
        key, w_key = jax.random.split( key)

        # initializing the parameters
        self.weights = jax.random.normal(w_key , shape = (in_features , out_features)) * 0.01
        self.bias = jnp.zeros((out_features,))
    
    def __call__(self , x ):
         return (jnp.dot( x , self.weights)) + self.bias


class SiLU:
    def __call__(self, x):
        sigmoid = 1 / (1 + jnp.exp(-x))
        return x * sigmoid

class Softmax:
    def __call__(self, x):
        x = x - jnp.max(x)
        e_x = jnp.exp(x)
        return e_x / jnp.sum(e_x , axis = -1 , keepdims=True)
        

class SwiGLU:

    def __init__(self , * , in_features , hidden_dims , key):
        k1 , k2 , k3 = jax.random.split(key , 3)
        self.in_features = in_features
        self.hidden_dims = hidden_dims
        self.silu = SiLU()
        self.gate = Linear(in_features=in_features , out_features=hidden_dims , key=k1)
        self.up = Linear(in_features=in_features , out_features = hidden_dims , key=k2 )
        self.down = Linear(in_features= hidden_dims , out_features= in_features , key=k3 )

    def __call__(self , x):
        
        # Gated Layer
        x_1 = self.gate(x)
        x_1 = self.silu(x_1)
        # Up layer
        x_2 = self.up(x)

        #Residual connetion
        x = x_1 * x_2
        x = self.down(x)

        return x
        

class SelfAttention:

    def __init__(self , embed_dims , key):
        k1 , k2 , k3 , k4 = jax.random.split(key , 4)        
        self.embed_dims = embed_dims
        self.softmax = Softmax()
        self.Wq = Linear(in_features=embed_dims , out_features=embed_dims , key= k1)
        self.Wk = Linear(in_features=embed_dims , out_features=embed_dims , key= k2)
        self.Wv = Linear(in_features=embed_dims , out_features=embed_dims , key= k3)
        self.Wo = Linear(in_features=embed_dims , out_features=embed_dims , key= k4)

    def __call__(self , x):

        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        K_T  = jnp.swapaxes(K , -1 , -2)
        scores = Q  @ K_T   
        scores = scores / jnp.sqrt(self.embed_dims)

        weights = self.softmax(scores)
        out = weights @ V

        return self.Wo(out)
    

        