// Persistent stdin/stdout ONNX runner for Xenova DeBERTa NLI.
// One JSON object per input line: {input_ids: number[][], attention_mask: number[][]}.
const readline = require('readline');
const ort = require('onnxruntime-node');

const modelPath = process.argv[2];
if (!modelPath) throw new Error('usage: node run_xenova_nli_onnx.js /path/to/model.onnx');

(async () => {
  const session = await ort.InferenceSession.create(modelPath);
  const lineReader = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
  for await (const line of lineReader) {
    if (!line.trim()) continue;
    try {
      const request = JSON.parse(line);
      const batch = request.input_ids.length;
      const length = request.input_ids[0].length;
      const ids = new BigInt64Array(request.input_ids.flat().map(BigInt));
      const mask = new BigInt64Array(request.attention_mask.flat().map(BigInt));
      const result = await session.run({
        input_ids: new ort.Tensor('int64', ids, [batch, length]),
        attention_mask: new ort.Tensor('int64', mask, [batch, length]),
      });
      process.stdout.write(JSON.stringify({ logits: Array.from(result.logits.data), batch }) + '\n');
    } catch (error) {
      process.stdout.write(JSON.stringify({ error: String(error && error.stack || error) }) + '\n');
    }
  }
})().catch((error) => { console.error(error); process.exit(1); });
