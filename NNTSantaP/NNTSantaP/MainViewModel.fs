namespace ViewModels


open System
open System.Threading
open System.Collections.ObjectModel
open System.Windows
open System.Windows.Input
open FsXaml
open FSharp.ViewModule
open FSharp.Charting
open FSharp.Charting.ChartTypes
open System.Windows
open System.Windows.Forms.Integration
open System.Threading.Tasks
open FSharp.Collections.ParallelSeq
open FSharp.Control
open FSharp.Control.Reactive


[<AutoOpen>]
module Utilities =
    let rand = new Random(int DateTime.Now.Ticks)

    type RandMessage =
        | GetMap of int * AsyncReplyChannel<(float * float)[]>
        | SetMap of int * AsyncReplyChannel<(float * float)[]>

    let mapAgent =
        MailboxProcessor.Start(fun inbox ->
            let random = new Random(int DateTime.Now.Ticks)
            let initMap n = Array.init n (fun _ -> random.Next(1001) |> float, random.Next(1001) |> float)
            let rec loop map =
                async {
                    let! msg = inbox.Receive()
                    match msg with
                    | GetMap(n, reply) ->
                        let map =
                            match map with
                            | [||] -> initMap n
                            | _ -> map
                        reply.Reply(map)
                        return! loop map
                    | SetMap(n, reply) ->
                        let map = initMap n
                        reply.Reply(map)
                        return! loop map
                }
            loop (initMap 0))

module ActivationFunction =
    ///                2
    /// f(x) = ------------------ - 1
    ///        1 + exp(-alpha * x)
    let bipolarSigmoidFunction (alpha:float) n = (2. / ( 1. + exp( -alpha * n ))) - 1.

    ///           2 * alpha * exp(-alpha * x )
    /// f'(x) = -------------------------------- = alpha * (1 - f(x)^2) / 2
    ///           (1 + exp(-alpha * x))^2
    let bipolarSigmoidDerivative (alpha:float) n =
        let y = bipolarSigmoidFunction alpha n
        alpha * ( 1. - y * y ) / 2.

    ///                1
    /// f(x) = ------------------
    ///        1 + exp(-alpha * x)
    let sigmoidFunction (alpha:float) n = 1. / ( 1. + exp( -alpha * n ))

    ///           alpha * exp(-alpha * x )
    /// f'(x) = ---------------------------- = alpha * f(x) * (1 - f(x))
    ///           (1 + exp(-alpha * x))^2
    let sigmoidDerivative (alpha:float) n =
        let y = sigmoidFunction alpha n
        alpha * y * (1. - y)

    /// f(x) = 1, if x >= 0, otherwise 0
    let thresholdFunction n = if n >= 0 then 1 else 0

type Activation = { Function : float -> float -> float
                    Derivative : float -> float -> float
                    RandomWeight : unit -> float }

[<CompilationRepresentation(CompilationRepresentationFlags.ModuleSuffix )>]
module Activation =
    let rand = new Random(int DateTime.Now.Ticks)

    let public sigmoidActivation =
       { Function =ActivationFunction.sigmoidFunction
         Derivative = ActivationFunction.sigmoidDerivative
         RandomWeight = (fun () -> rand.NextDouble())    }

    let public bipolarSigmoidActivation =
       { Function =ActivationFunction.bipolarSigmoidFunction
         Derivative = ActivationFunction.bipolarSigmoidDerivative
         RandomWeight = (fun () -> rand.NextDouble())    }

//[<Struct>]
type Neuron =
    { inputsCount : int
      output:float
      threshold:float
      weights : float array }
    member this.item n = this.weights |> Array.item n

[<CompilationRepresentation(CompilationRepresentationFlags.ModuleSuffix )>]
module Neuron =
    let [<Literal>] private rangeLen = 1000.

    let create (inputs : int) =
        let inputs = max 1 inputs
        { inputsCount = inputs
          threshold = rand.NextDouble() * rangeLen
          output = 0.
          weights = Array.init inputs (fun _ -> rand.NextDouble() * rangeLen  ) }

    let compute (neuron : Neuron) (input : float array) =
        let weigths = neuron.weights
        [ 0..neuron.inputsCount - 1 ] |> Seq.fold (fun s i -> s + abs (weigths.[i] - input.[i])) 0.

    let computeActivation (neuron : Neuron) (input:float array) alpha (activationF:float -> float -> float) =
        let total = neuron.weights |> Array.zip(input) |> Array.sumBy(fun (a,b) -> a * b)
        let total = total + neuron.threshold
        let output = activationF alpha total
        { neuron with output = output}

/// A layer represents a collection of neurons
//[<Struct>]
type Layer =
    {
      neuronsCount : int
      inputsCount : int
      neurons : Neuron array
      output : float array
    }
    member this.item n = this.neurons |> Array.item n

[<CompilationRepresentation(CompilationRepresentationFlags.ModuleSuffix)>]
module Layer =
    let create neuronsCount inputsCount =
        let neuronsCount = max 1 neuronsCount
        { neuronsCount = neuronsCount
          inputsCount = inputsCount
          #if PARALLEL
          neurons = Array.Parallel.init neuronsCount (fun i -> Neuron.create inputsCount)
          #else
          neurons = Array.init neuronsCount (fun i -> Neuron.create inputsCount)
          #endif
          output = Array.zeroCreate<float> neuronsCount }

    /// Compute output vector of the layer
    let compute (inputs : float array) (layer : Layer) =
        let neuronsCount = layer.neuronsCount

        #if PARALLEL
        let output = Array.Parallel.init neuronsCount (fun i -> Neuron.compute layer.neurons.[i] inputs)
        #else
        let output = Array.init neuronsCount (fun i -> Neuron.compute layer.neurons.[i] inputs)
        #endif

        { layer with output = output }

    let getLayerOutputs layer =
        seq{ for neuron in layer.neurons do yield neuron.output}

type Connection = { toNeuron:Neuron
                    fromNeuron:Neuron
                    weight:float }

[<CompilationRepresentation(CompilationRepresentationFlags.ModuleSuffix )>]
module Connection =
    let create toNeuron fromNeuron =
        {
            toNeuron = toNeuron
            fromNeuron = fromNeuron
            weight = rand.NextDouble() * 0.5
        }

    let connectLayers (nextLayer:Layer) (neurons:Neuron array) = seq {
        for neuron in neurons do
            for layerNeuron in nextLayer.neurons do
                yield create neuron layerNeuron }


//[<Struct>]
type Network =
    {
      inputsCount : int
      layersCount : int
      layers : Layer array
      ouputLayer : Layer
      activation : Activation
      output : float array
    }
    member this.item n = this.layers |> Array.item n

[<CompilationRepresentation(CompilationRepresentationFlags.ModuleSuffix)>]
module Network =
    let create inputsCount layersCount =
        let layers = Array.init layersCount (fun _ -> Layer.create layersCount inputsCount)
        {
          inputsCount = inputsCount
          layersCount = layersCount
          layers = layers
          ouputLayer = layers |> Array.last
          activation = Activation.sigmoidActivation
          output = [||]
        }

    let computeLayers (network : Network) (input : float array) =
        #if PARALLEL
        let layers = network.layers |> Array.Parallel.map(Layer.compute input)
        #else
        let layers = network.layers |> Array.map(Layer.compute input)
        #endif
        { network with layers = layers; ouputLayer = layers |> Array.last ; output = (layers |> Array.last).output }

    let computeOutputLayer (network : Network) (input : float array) =
        let layer = network.ouputLayer |> Layer.compute input
        { network with ouputLayer = layer ; output = layer.output }

    let foundBestOutput (network : Network) =
        let output = network.ouputLayer.output |> Array.toList
        [ 0..output.Length - 1 ]
        |> Seq.zip (output)
        |> Seq.minBy (fun (n, _) -> n)
        |> snd


//    [<Struct>]
type ElasticNetworkLearning =
    { learningRate : float
      learningRadius : float
      squaredRadius : float
      distance : float array
      network : Network }


[<CompilationRepresentation(CompilationRepresentationFlags.ModuleSuffix)>]
module NetworkLearning =
    let create (network : Network) =
        let neuronsCount = network.ouputLayer.neuronsCount
        let delta = Math.PI * 2.0 / (float neuronsCount)

        let rec initDistance i alpha acc =
            match i with
            | n when n < neuronsCount ->
                let x = 0.5 * Math.Cos(alpha) - 0.5
                let y = 0.5 * Math.Sin(alpha)
                initDistance (i + 1) (alpha + delta) ((x * x + y * y)::acc)
            | _ -> acc |> List.toArray
        // initial arbitrary values
        { learningRate = 0.1
          learningRadius = 0.5
          squaredRadius = 98.
          distance = initDistance 0 delta []
          network = network }


    let setLearningRate learningRate (learning : ElasticNetworkLearning) =
        { learning with learningRate = max 0. (min 1. learningRate) }

    let setLearningRadius learningRadius (learning : ElasticNetworkLearning) =
        let learningRadius = max 0. (min 1. learningRadius)
        { learning with learningRadius = learningRadius
                        squaredRadius = 2. * learningRadius * learningRadius }

    let compute (learning : ElasticNetworkLearning) (input : float array) =
        let learningRate = learning.learningRate
        let network = Network.computeOutputLayer learning.network input
        let bestNetwork = Network.foundBestOutput network
        let layer = network.ouputLayer

        #if PARALLEL
        System.Threading.Tasks.Parallel.For(0, layer.neuronsCount - 1, fun j ->
        #else
        for j = 0 to layer.neuronsCount - 1 do
        #endif
            let neuron = layer.item j
            let factor = exp (-learning.distance.[abs (j - bestNetwork)] / learning.squaredRadius)
            for i = 0 to neuron.inputsCount - 1 do
                let e = (input.[i] - neuron.item i) * factor
                neuron.weights.[i] <- neuron.weights.[i] + (e + learningRate)
        #if PARALLEL
            ) |> ignore
        #endif
        { learning with network = network }


module TravelingSantaProblem =

    type TravelingSantaProblem(neurons:int, learningRate:float, cities:(float*float)[]) =

        let asyncSeq = FSharp.Control.AsyncSeq.AsyncSeqBuilder()

        let foundBestPath iterations = asyncSeq {
            let network = Network.create 2 neurons
            let trainer = NetworkLearning.create network
            let fixedLearningRate = learningRate / 20.
            let driftingLearningRate = fixedLearningRate * 19.
            let input = Array.zeroCreate<float> 2
            let iterations = float iterations
            let citiesCount = cities.Length
            let lenNeurons = neurons
            let path = Array.zeroCreate<(float * float)> (lenNeurons + 1)

            let getNeuronWeight (trainer:ElasticNetworkLearning) n w =
                (trainer.network.ouputLayer.item n).item w

            for i = 0 to (int iterations - 1) do

                let learningRateUpdated = driftingLearningRate * (iterations - float i) / iterations + fixedLearningRate
                let trainer = NetworkLearning.setLearningRate learningRateUpdated trainer
                let learningRadiusUpdated = trainer.learningRadius * (iterations - float i) / iterations
                let trainer = NetworkLearning.setLearningRadius learningRadiusUpdated trainer

                let currentStep = rand.Next(citiesCount)
                input.[0] <- cities.[currentStep] |> fst
                input.[1] <- cities.[currentStep] |> snd
                let trainer = NetworkLearning.compute trainer input

                #if PARALLEL
                let path = Array.Parallel.init (lenNeurons) (
                                          fun j -> if j = lenNeurons - 1
                                                   then ((trainer.network.item 0).item 0).item 0, ((trainer.network.item 0).item 0).item 1
                                                   else ((trainer.network.item 0).item j).item 0, ((trainer.network.item 0).item j).item 1)
                #else

                let getNeuronWeight = getNeuronWeight trainer

                for j = 0 to lenNeurons - 1 do
                    path.[j] <- getNeuronWeight j 0 , getNeuronWeight j 1
                path.[lenNeurons] <-  getNeuronWeight 0 0, getNeuronWeight 0 1

                #endif


                if i % 100 = 0 then
                    yield (i - 1 , path)
                    do! Async.Sleep 5
                yield ((int iterations - 1), path)
                }

        member this.Execute = foundBestPath

open TravelingSantaProblem

type MainViewModel() as this =
    inherit ViewModelBase()


    let mutable cts = new CancellationTokenSource()

    let pathStream = Event<(float * float)[]>()
    let pathObs =
        pathStream.Publish |> Observable.map(id)

    let pointsStream = Event<(float * float)[]>()

    let pointsObs =
        pointsStream.Publish |> Observable.map id


    let cities = this.Factory.Backing(<@ this.Cities @>, 100)
    let iterations = this.Factory.Backing(<@ this.Iterations @>, 25000)

    let neurons = this.Factory.Backing(<@ this.Neurons @>, 80)

    let learningRate = this.Factory.Backing(<@ this.LearningRate @>, 0.5)

    let currentIterations = this.Factory.Backing(<@ this.CurrentIterations @>, 0)
    let executionTime = this.Factory.Backing(<@ this.ExecutionTime @>, "")
    do mapAgent.PostAndReply(fun ch -> SetMap(cities.Value, ch)) |> pointsStream.Trigger

    let livePathChart = LiveChart.Line(pathObs)
    let livePointsChart = LiveChart.Point(pointsObs)

    let chartCombine = Chart.Combine([livePointsChart; livePathChart]).WithYAxis(Enabled=false).WithXAxis(Enabled=false)
    let chart = new ChartControl(chartCombine)
    let host = new WindowsFormsHost(Child = chart)
    let hostChart = this.Factory.Backing(<@ this.Chart @>, host)

    let initControls n =
        this.CurrentIterations <- 0
        this.ExecutionTime <- ""
        pointsStream.Trigger [||]
        mapAgent.PostAndReply(fun ch -> SetMap(n, ch)) |> pointsStream.Trigger

    let updateCtrl ui i (points : (float * float)[]) = async {
        do! Async.SwitchToContext ui
        this.CurrentIterations <- (i + 1)
        pathStream.Trigger points
        }

    let onCancel _ =
        this.CurrentIterations <- 0
        this.ExecutionTime <- ""
        pathStream.Trigger [||]
        cts.Dispose()
        cts <- new CancellationTokenSource()
        this.StartCommand.CancellationToken <- cts.Token

    let cancelClear () =
        cts.Cancel()

    let cancel =
        this.Factory.CommandSyncChecked(cancelClear, (fun _ -> this.OperationExecuting), [ <@@ this.OperationExecuting @@> ])

    let initPoints =
        this.Factory.CommandSyncParamChecked(initControls, (fun _ -> not this.OperationExecuting), [ <@@ this.OperationExecuting @@> ])

    let start =
        this.Factory.CommandAsync((fun ui -> async { let time = System.Diagnostics.Stopwatch.StartNew()
                                                     let updateControl = updateCtrl ui
                                                     let! cities = mapAgent.PostAndAsyncReply(fun ch -> GetMap(cities.Value, ch))

                                                     let tsp = TravelingSantaProblem(neurons.Value,learningRate.Value,cities)
                                                     for (i, path) in tsp.Execute (iterations.Value) do
                                                           do! updateControl i path
                                                     this.ExecutionTime <- sprintf "Time %d ms" time.ElapsedMilliseconds}), token=cts.Token, onCancel=onCancel)
    do initControls (cities.Value)

    member this.Chart
        with get () = hostChart.Value
        and set value = hostChart.Value <- value

    member this.Cities
        with get () = cities.Value
        and set value = cities.Value <- value

    member this.Neurons
        with get () = neurons.Value
        and set value = neurons.Value <- value

    member this.LearningRate
        with get () = learningRate.Value
        and set value = learningRate.Value <- value

    member this.Iterations
        with get () = iterations.Value
        and set value = iterations.Value <- value

    member this.CurrentIterations
        with get () = currentIterations.Value
        and set value = currentIterations.Value <- value

    member this.ExecutionTime
        with get () = executionTime.Value
        and set value = executionTime.Value <- value

    member this.InitPointsCommand = initPoints
    member this.StartCommand : IAsyncNotifyCommand = start
    member this.CancelCommand = cancel


