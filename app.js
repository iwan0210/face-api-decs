require('dotenv').config()
require('@tensorflow/tfjs-node')
const express = require('express')
const faceapi = require('face-api.js')
const canvas = require('canvas')
const path = require('path')

const {
    Canvas,
    Image,
    ImageData
} = canvas

faceapi.env.monkeyPatch({
    Canvas,
    Image,
    ImageData
})

const app = express();

app.set('trust proxy', true)

app.use(express.json({
    limit: '10mb'
}))

const API_TOKEN = process.env.API_TOKEN || 'my-secret-token';
const PORT = process.env.PORT || 3000;
const MODEL_PATH = path.join(__dirname, 'models')

let modelsLoaded = false

async function loadModels() {
    console.log('Loading models...')

    await faceapi.nets.ssdMobilenetv1.loadFromDisk(
        path.join(MODEL_PATH)
    )

    await faceapi.nets.faceLandmark68Net.loadFromDisk(
        path.join(MODEL_PATH)
    )

    await faceapi.nets.faceRecognitionNet.loadFromDisk(
        path.join(MODEL_PATH)
    )

    modelsLoaded = true

    console.log('Models loaded')
}

function base64ToBuffer(base64) {
    // Supports:
    // data:image/jpeg;base64,...
    // data:image/png;base64,...
    // or pure Base64

    if (base64.includes(',')) {
        base64 = base64.split(',')[1]
    }

    return Buffer.from(base64, 'base64')
}

function authenticateFaceApi(req, res, next) {
    const token = req.get('X-API-Token')

    if (token !== API_TOKEN) {
        return res.status(401).json({
            success: false,
            message: 'Unauthorized'
        })
    }

    next()
}

app.post('/api/face/descriptor',
    authenticateFaceApi,
    async (req, res) => {

        const startTime = process.hrtime.bigint()

        try {
            if (!modelsLoaded) {
                return res.status(503).json({
                    success: false,
                    message: 'Face recognition models are not ready'
                })
            }

            const { image } = req.body

            if (!image) {
                return res.status(400).json({
                    success: false,
                    message: 'image is required'
                })
            }

            const imageBuffer = base64ToBuffer(image)

            const img = await canvas.loadImage(imageBuffer)

            const detection = await faceapi
                .detectSingleFace(img)
                .withFaceLandmarks()
                .withFaceDescriptor()

            const processTimeSeconds =
                Number(process.hrtime.bigint() - startTime) / 1_000_000_000

            if (!detection) {
                return res.status(200).json({
                    success: true,
                    detected: false,
                    message: 'No face detected',
                    processTimeSeconds: Number(processTimeSeconds.toFixed(3))
                })
            }

            return res.json({
                success: true,
                detected: true,
                score: detection.detection.score,
                descriptor: Array.from(detection.descriptor),
                box: {
                    x: detection.detection.box.x,
                    y: detection.detection.box.y,
                    width: detection.detection.box.width,
                    height: detection.detection.box.height
                },
                processTimeSeconds: Number(processTimeSeconds.toFixed(3))
            })

        } catch (error) {
            console.error(error)

            return res.status(500).json({
                success: false,
                message: error.message
            })
        }
    })

app.get('/api/health', (req, res) => {
    res.json({
        success: true,
        modelsLoaded
    })
})

async function start() {
    await loadModels();

    app.listen(PORT, '0.0.0.0', () => {
        console.log(`Face service running on port ${PORT}`)
    })
}

start().catch(error => {
    console.error('Failed to start:', error)
    process.exit(1)
})