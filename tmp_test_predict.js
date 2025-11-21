const http = require('http');
const data = JSON.stringify({
  weather: { main: { temp: 25, humidity: 78 }, name: 'Test' },
  sensorData: [ { zone: 'A', soilMoisture: 45, temperature: 25, nutrients: '10-7-8' } ]
});
const options = {
  hostname: 'localhost',
  port: 5000,
  path: '/api/predict-insights',
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Content-Length': Buffer.byteLength(data)
  }
};
const req = http.request(options, (res) => {
  let body = '';
  res.on('data', (chunk) => body += chunk);
  res.on('end', () => console.log('Response:', body));
});
req.on('error', (e) => console.error('Request error', e));
req.write(data);
req.end();
