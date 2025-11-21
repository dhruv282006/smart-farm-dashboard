import sys
import json
import argparse
import os
import joblib


def main():
    parser = argparse.ArgumentParser(description='Predict yield from rainfall, temperature, humidity.')
    parser.add_argument('--rainfall', type=float)
    parser.add_argument('--temperature', type=float)
    parser.add_argument('--humidity', type=float)
    parser.add_argument('--model', type=str, default=None, help='Path to model file (joblib)')
    parser.add_argument('--json', type=str, default=None, help='JSON string with keys rainfall,temperature,humidity')
    args = parser.parse_args()

    try:
        if args.json:
            data = json.loads(args.json)
            rainfall = float(data.get('rainfall'))
            temperature = float(data.get('temperature'))
            humidity = float(data.get('humidity'))
        else:
            if args.rainfall is None or args.temperature is None or args.humidity is None:
                print(json.dumps({'error': 'Please provide rainfall, temperature and humidity via flags or --json'}))
                return
            rainfall = args.rainfall
            temperature = args.temperature
            humidity = args.humidity

        model_path = args.model or 'yield_model_v1_latest.joblib'
        if not os.path.exists(model_path):
            # try to find recent joblib in current dir
            candidates = [f for f in os.listdir('.') if f.startswith('yield_model') and f.endswith('.joblib')]
            if candidates:
                candidates.sort()
                model_path = candidates[-1]
        model = joblib.load(model_path)

        pred = model.predict([[rainfall, temperature, humidity]])
        out = {'predicted_yield': round(float(pred[0]), 3), 'model_used': model_path}
        print(json.dumps(out))
    except Exception as e:
        print(json.dumps({'error': str(e)}))


if __name__ == '__main__':
    main()
