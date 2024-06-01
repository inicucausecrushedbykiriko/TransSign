import argparse
from scripts import preprocess_videos, train_model, translate_video

def main():
    parser = argparse.ArgumentParser(description='Sign Language Translator')
    parser.add_argument('--preprocess', action='store_true', help='Preprocess videos')
    parser.add_argument('--train', action='store_true', help='Train model')
    parser.add_argument('--translate', action='store_true', help='Translate video')
    parser.add_argument('--input_video', type=str, help='Path to input video')
    parser.add_argument('--output_video', type=str, help='Path to output video')
    parser.add_argument('--direction', type=str, choices=['asl_to_csl', 'csl_to_asl'], help='Translation direction')

    args = parser.parse_args()

    if args.preprocess:
        preprocess_videos.preprocess()
    elif args.train:
        train_model.train()
    elif args.translate:
        translate_video.translate(args.input_video, args.output_video, args.direction)
    else:
        print("No valid arguments provided. Use --help for more information.")

if __name__ == "__main__":
    main()

