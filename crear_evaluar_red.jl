## Este file crea la estructura de la red y además la evalua

#Estos son 
using Random
using LinearAlgebra
using MLDatasets

#Definimos nuestro tipo de dato y le decimos exactamente que son cada cosa
struct Red
    num_layers::Int64
    tamanos::Vector{Int16}
    biases::Vector{Array{Float64}}
    weights::Vector{Array{Float64}}
end

#Esta función crea la red
function crea_red(tamanos)
    num=length(tamanos)

    #menos el ultimo
    a = view(tamanos, 1:(num-1))
    #menos el primero
    b = view( tamanos,  2:num)
    
    biases=[randn( Float64,  (i,1) ) for i in b ]
    weights=[randn( Float64, (i,j) ) for (i,j) in zip( b, a ) ]
    
    return Red(num,tamanos,biases,weights)
end

sigma(x)=1/(1+exp(-x))

# esta funciòn hace el feedforward
function feedforward(red::Red,x0)
    a=x0        
    for  i in collect(1:red.num_layers-1)        
        z=red.weights[i]*a+red.biases[i]
        a=sigma.(z)        
    end    
    return a    
end


function carga_datos(n::Int64)
    x , y = MNIST.traindata() #Cargamos los datos TODOS
    datos = [] #regresaremos esos n datos en un vector de vectores para manejarlos más fácil
    etiquetas = [] #regresaremos esas n etiquetas en un vector de vectores para manejarlos más fácil
    for i in 1:n
        push!(datos,  reshape(x[:,:,i], 784) )
        push!(etiquetas,   vectorizar( y[i] ) )
    end
    return datos,etiquetas
end

function vectorizar(valor::Int64)
    vector =  zeros(Int16, 10)
    vector[valor+1] = 1
    return vector
end




#dada la salida de la red la vuelve un vector con un 1 en el máximo y 0 en todos los demas 
function Target(a2)
    A2=zeros(length(a2))
    valor,pos=findmax(a2)
    A2[pos]=1.0
    return A2
end

#compara la salida de la red vs lo que debe de ser y regresa 1 si son iguales
function Exito(T,a2)
    e=sum(abs.(T.-Target(a2)))
    if e==0
        return 1
    else
        return 0
    end
end

#funcion que regresa el accuracy de la red 
function accuracy(T,a2)
        acum=0.0
        for j in collect(1:size(T)[1])
            acum+=Exito(T[j],a2[j])
        end
        return acum/size(T)[1]
end
data,etiquetas=carga_datos(10)#cargamos la cantidad de datos que querramos
tamanos = [784,30,10] #definimos la topología de nuestra red
red = crea_red(tamanos) #generamos la red

A=[] #un vector auxiliar que nos va a aydar a que todo sea más fácil
for i in collect(1:length(data))#calculamos y guardamos la salida de nuestra red
    push!(A,feedforward(red,data[i]))
end
println("la red hace la tarea de forma exitosa con esta tasa:", accuracy(etiquetas,A))
