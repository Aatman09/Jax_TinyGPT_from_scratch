import jax.numpy as jnp 
import jax
import optax

# for testing
key = jax.random.PRNGKey(0)

class RMSnorm:
    def __init__(self ,dim : int , epsilon : float = 1e-6):
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
    def __init__(self ,* , in_features :int , out_features :int , key ):
        
        self.in_features = in_features
        self.out_features = out_features 
        key, w_key = jax.random.split( key)

        # initializing the parameters
        self.weights = jax.random.normal(w_key , shape = (in_features , out_features)) * 0.01
        self.bias = jnp.zeros((out_features,))
    
    def __call__(self , x ):
         return (jnp.matmul( x , self.weights)) + self.bias


class SiLU:
    def __call__(self, x):
        sigmoid = 1 / (1 + jnp.exp(-x))
        return x * sigmoid

class Softmax:
    def __call__(self, x):
        x = x - jnp.max(x, axis=-1, keepdims=True)
        e_x = jnp.exp(x)
        return e_x / jnp.sum(e_x , axis = -1 , keepdims=True)
        

class SwiGLU:

    def __init__(self , * , in_features :int, hidden_dims :int , key):
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

    def __init__(self , embed_dims : int, key):
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

        T = x.shape[1]
        mask = jnp.tril(jnp.ones((T, T)))
        scores = scores * mask - 1e9 * (1 - mask)

        weights = self.softmax(scores)
        out = weights @ V

        return self.Wo(out)
    
class TransformerBlock:
    def __init__(self , dim , hidden , key):
        k1 , k2 = jax.random.split(key , 2)
        self.attention = SelfAttention(dim , k1)
        self.mlp =  SwiGLU(in_features= dim , hidden_dims= hidden , key = k2)
        self.norm1 = RMSnorm(dim = dim)
        self.norm2  = RMSnorm(dim = dim)

    def __call__(self , x):

        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x

class TinyGPT:
    def __init__(self , vocab_size , dim , hidden , num_layers , key):

        self.embed = jax.random.normal(key , (vocab_size ,dim)) * 0.01

        self.layers = []

        for i in range (num_layers) :
            key , k = jax.random.split(key)
            self.layers.append(TransformerBlock(dim , hidden , k ))

        self.norm  = RMSnorm(dim)
        self.linear = Linear(in_features=dim , out_features= vocab_size , key =key)

    def __call__(self , tokens):

        x = self.embed[tokens]
        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        logits = self.linear(x)

        return logits


def loss_function(model , tokens):
    logits = model(tokens[:,:-1])
    targets = tokens[: , 1:]
    loss = optax.softmax_cross_entropy_with_integer_labels(logits , targets).mean()
    return loss

@jax.jit
def train_step(model , opt_state , tokens):
    loss , grads = jax.value_and_grad(loss_function)(model , tokens)
    updates , opt_state = optimizer.update(grads , opt_state)
    model = optax.apply_updates(model , updates)
    return model, opt_state , loss


vocab_size = 50
dim = 32 
hidden = 64 
num_layers = 2
key = jax.random.PRNGKey(0)

model = TinyGPT(vocab_size , dim , hidden , num_layers , key)

optimizer = optax.adam(1e-6)
opt_state = optimizer.init(model)

#fake dataset
tokens = jax.random.randint(key , (4,10) , 0 , vocab_size)


for step in range(200):
    model , opt_state , loss = train_step(model , opt_state  , tokens)
    if step % 20 == 0 :
        print(step , loss)