 
 // Some general parameters
 const BATCH_SIZE = 16;
 const SEQ_LEN = 5;
 const LSTM_UNITS = 50;
 const EPOCHS = 50;
 const model = tf.sequential();
 
 function* dataGenerator() {
   const numElements = 10;
   let index = 0;
   while (index < numElements) {
     index++;
     let x = tf.randomUniform([BATCH_SIZE, SEQ_LEN, 1]);
	 let y = tf.oneHot(tf.argMax(x, 1).squeeze(), SEQ_LEN);
	 yield {xs: x, ys: y};
   }
}

const ds = tf.data.generator(dataGenerator).repeat();

async function createModel() {
 // Add layers for our model
	model.add(
	  tf.layers.lstm({
		units: LSTM_UNITS,
		returnSequences: false,
		batchInputShape: [null, SEQ_LEN, 1],
		recurrentActivation: 'tanh'
	  })
	);
	model.add(tf.layers.batchNormalization());
	model.add(
	  tf.layers.dense({
		units: SEQ_LEN,
		activation: 'softmax'
	  })
	);
	  
	// Simplified Keras syntax
	model.compile({
		loss: tf.losses.meanSquaredError,
		optimizer: tf.train.adam(),
		metrics: ['accuracy']
	});
	
 }
 
 async function train() {
 
	// Create the model
	await createModel();
	
	// Train
	await model.fitDataset(ds, {
		batchesPerEpoch: 10,
		epochs: EPOCHS,
		callbacks: tfvis.show.fitCallbacks(
		  { name: 'Training Performance' },
		  ['loss', 'loss', 'accuracy', 'acc'], 
		  { height: 200, callbacks: ['onEpochEnd'] }
		)
	  });
 }
 
 async function runDemo() {
	document.getElementById('trainModel').addEventListener('click', async () => {
		await train();
	});
	
	document.getElementById('testModel').addEventListener('click', async () => {
		tf.tidy(() => {
			const data = tf.randomUniform([1, SEQ_LEN, 1]);
			const prediction = tf.argMax(model.predict(data), 1);
			
			document.getElementById('testInput').textContent = data;
			document.getElementById('testOutput').textContent = data.dataSync()[prediction.dataSync()];
		});
	});
}
 
 runDemo();