const express = require('express');
const cors = require('cors');
const fetch = require('node-fetch');
const multer = require('multer');
const { OpenAI } = require('openai');
const path = require('path');
require('dotenv').config();
const axios = require('axios');

// Initialize OpenAI client
const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY
});

// Debug: Check if .env is loaded
console.log('Environment check on startup:', {
    hasHuggingFaceKey: !!process.env.HUGGINGFACE_API_KEY,
    hasOpenAIKey: !!process.env.OPENAI_API_KEY,
    envPath: path.join(__dirname, '.env')
});

const app = express();
const PORT = process.env.PORT || 3001;

// Middleware
app.use(cors());
app.use(express.json({ limit: '50mb' }));

// Root endpoint
app.get('/', (req, res) => {
    res.json({ message: 'Architecture Critique Analyzer API is running!' });
});

// Test endpoint
app.get('/test', (req, res) => {
    res.json({ message: 'Server is running!' });
});

// Architectural terms for better classification
const architecturalTerms = {
    styles: [
        'modern', 'contemporary', 'minimalist', 'brutalist', 'postmodern',
        'art deco', 'gothic', 'classical', 'neoclassical', 'romanesque',
        'baroque', 'victorian', 'industrial', 'scandinavian', 'japanese',
        'traditional', 'vernacular', 'high-tech', 'deconstructivist', 'organic'
    ],
    elements: [
        'facade', 'roof', 'wall', 'window', 'door', 'column', 'beam',
        'arch', 'dome', 'vault', 'staircase', 'balcony', 'terrace',
        'courtyard', 'atrium', 'skylight', 'canopy', 'overhang',
        'entrance', 'exit', 'ramp', 'elevator', 'escalator', 'bridge',
        'tower', 'spire', 'minaret', 'gallery', 'corridor', 'passage'
    ],
    materials: [
        'concrete', 'steel', 'glass', 'wood', 'stone', 'brick', 'metal',
        'aluminum', 'copper', 'bronze', 'marble', 'granite', 'timber',
        'composite', 'ceramic', 'terracotta', 'corten steel', 'zinc',
        'titanium', 'gold', 'silver', 'plaster', 'stucco', 'masonry'
    ],
    features: [
        'sustainable', 'green roof', 'solar panels', 'rainwater harvesting',
        'natural ventilation', 'daylighting', 'thermal mass', 'passive design',
        'adaptive reuse', 'modular', 'prefabricated', 'parametric design',
        'smart building', 'biophilic design', 'zero energy', 'net zero',
        'LEED certified', 'energy efficient', 'water efficient', 'recycled materials'
    ],
    spaces: [
        'interior', 'exterior', 'public space', 'private space', 'circulation',
        'lobby', 'foyer', 'gallery', 'exhibition space', 'workshop',
        'studio', 'office', 'residential', 'commercial', 'cultural',
        'educational', 'institutional', 'religious', 'recreational', 'industrial',
        'retail', 'hospitality', 'transportation', 'infrastructure'
    ],
    characteristics: [
        'openness', 'transparency', 'fluidity', 'solidity', 'lightness',
        'heaviness', 'simplicity', 'complexity', 'symmetry', 'asymmetry',
        'rhythm', 'repetition', 'contrast', 'harmony', 'balance',
        'proportion', 'scale', 'hierarchy', 'unity', 'diversity'
    ],
    vibes: [
        'biophilic', 'organic', 'natural', 'artificial', 'industrial',
        'luxurious', 'minimal', 'cozy', 'grand', 'intimate',
        'dynamic', 'static', 'welcoming', 'imposing', 'serene',
        'energetic', 'calm', 'dramatic', 'subtle', 'bold'
    ]
};

// Hugging Face API endpoint
app.post('/analyze', multer().single('image'), async (req, res) => {
    try {
        const inputCritique = req.body.critique;
        const inputConcept = req.body.concept;
        const inputLanguage = req.body.language || 'en';

        if (!inputCritique) {
            return res.status(400).json({ error: 'No critique provided' });
        }

        // Get image tags if image is provided
        let imageTags = {
            styles: [],
            elements: [],
            materials: [],
            features: [],
            spaces: [],
            characteristics: [],
            vibes: [],
            detected: []
        };

        if (req.file) {
            const imageBuffer = req.file.buffer;
            const imageBase64 = imageBuffer.toString('base64');
            
            console.log('Sending image to Hugging Face API...');
            const imageResponse = await fetch(
                "https://api-inference.huggingface.co/models/microsoft/resnet-50",
                {
                    headers: {
                        "Authorization": `Bearer ${process.env.HUGGINGFACE_API_KEY}`,
                        "Content-Type": "application/json",
                    },
                    method: "POST",
                    body: JSON.stringify({
                        inputs: `data:image/jpeg;base64,${imageBase64}`
                    }),
                }
            );

            if (imageResponse.ok) {
                const imageResult = await imageResponse.json();
                console.log('Raw Hugging Face API Response:', JSON.stringify(imageResult, null, 2));
                
                // Extract tags from the image analysis
                const tags = Array.isArray(imageResult) ? imageResult.map(item => item.label.toLowerCase()) : [];
                console.log('Initial extracted tags:', tags);
                
                // Initialize imageTags with empty arrays
                imageTags = {
                    styles: [],
                    elements: [],
                    materials: [],
                    features: [],
                    spaces: [],
                    characteristics: [],
                    vibes: []
                };

                // Use OpenAI to analyze the image detection results and extract architectural terms
                try {
                    const tagResponse = await openai.chat.completions.create({
                        model: "gpt-4",
                        messages: [
                            {
                                role: "system",
                                content: `You are an architectural tag generator. Your task is to convert detected objects into architectural terms and make reasonable architectural inferences.

Available categories and terms: ${JSON.stringify(architecturalTerms)}

Rules:
1. Start with direct matches from detected objects
2. Make reasonable architectural inferences based on the detected objects
3. Each category should have at least 2-3 relevant terms
4. All terms must be logically connected to the detected objects
5. Return a JSON object with arrays for each category

Example:
Input: ["library", "prison", "spiral", "desk"]
Output: {
    "styles": ["institutional", "classical"],
    "elements": ["spiral", "column", "window"],
    "materials": ["stone", "concrete"],
    "features": ["furniture", "security"],
    "spaces": ["library", "institutional", "public space"],
    "characteristics": ["symmetry", "imposing"],
    "vibes": ["serene", "imposing"]
}`,
                            },
                            {
                                role: "user",
                                content: `Convert these detected objects into architectural terms: ${JSON.stringify(tags)}`
                            }
                        ],
                        temperature: 0.3,
                        max_tokens: 500
                    });

                    const extractedTerms = JSON.parse(tagResponse.choices[0].message.content);
                    
                    // Update imageTags with the extracted terms
                    Object.keys(extractedTerms).forEach(category => {
                        if (imageTags.hasOwnProperty(category)) {
                            imageTags[category] = [...new Set(extractedTerms[category])];
                        }
                    });
                } catch (error) {
                    console.error('Error extracting architectural terms:', error);
                }

                console.log('Final tags structure:', imageTags);
            }
        }

        // Get the analysis from OpenAI
        const analysis = await openai.chat.completions.create({
            model: "gpt-4",
            messages: [
                {
                    role: "system",
                    content: `You are an architectural critic and analyst. Provide a detailed analysis of the architectural image in ${inputLanguage === 'ko' ? 'Korean' : 'English'} in the following format:

${inputLanguage === 'ko' ? '비판 분석:' : 'CRITIQUE ANALYSIS:'}
[Provide a comprehensive analysis of the architectural elements, style, and design principles. Include:
1. Educational explanation of key architectural concepts mentioned in the critique
2. Historical context and notable architects associated with the style/concepts
3. Specific details about form, space, materials, and their relationships
Minimum 300 words.]

${inputLanguage === 'ko' ? '개선 계획:' : 'IMPROVEMENT PLAN:'}
[Provide detailed suggestions for improvements, numbered list. Each suggestion should include:
1. Specific examples and practical applications
2. References to similar architectural solutions
3. Educational context for the suggested improvements
Minimum 200 words.]

${inputLanguage === 'ko' ? '이미지와 텍스트 분석의 일치성:' : 'ALIGNMENT ANALYSIS:'}
[Compare the detected architectural elements from the image with the provided critique and concept. Discuss:
1. How well they align
2. Any discrepancies
3. Educational context for the architectural concepts
Minimum 150 words.]`
                },
                {
                    role: "user",
                    content: `Analyze this architectural image and provide a detailed critique and improvement plan.

Image Analysis:
${JSON.stringify(imageTags, null, 2)}

Text Input:
Critique: ${inputCritique}
Concept: ${inputConcept}`
                }
            ],
            temperature: 0.7,
            max_tokens: 3000
        });

        // Split the analysis into critique, improvements, and alignment
        const fullAnalysis = analysis.choices[0].message.content;
        const parts = fullAnalysis.split(/IMPROVEMENT PLAN:|개선 계획:|ALIGNMENT ANALYSIS:|이미지와 텍스트 분석의 일치성:/);
        const analysisCritique = parts[0].replace(/CRITIQUE ANALYSIS:|비판 분석:/, '').trim();
        const analysisImprovements = parts[1] ? parts[1].trim() : '';
        const analysisAlignment = parts[2] ? parts[2].trim() : '';

        const response = {
            tags: imageTags,
            critique: analysisCritique,
            improvements: analysisImprovements,
            alignment: analysisAlignment
        };

        // Get references based on critique and tags
        try {
            const references = await getReferences(inputCritique, imageTags, inputLanguage);
            response.references = references;
        } catch (error) {
            console.error('Error getting references:', error);
            response.references = {
                title: inputLanguage === 'ko' ? "참고 자료" : "References",
                items: []
            };
        }

        console.log('Sending response:', response);
        res.json(response);

    } catch (error) {
        console.error('Error:', error);
        res.status(500).json({ error: error.message });
    }
});

async function getReferences(critique, tags, language) {
    const references = {
        title: language === 'ko' ? "참고 자료" : "References",
        items: []
    };

    try {
        // Get references using OpenAI
        const referencesResponse = await openai.chat.completions.create({
            model: "gpt-4",
            messages: [
                {
                    role: "system",
                    content: `You are an architectural reference expert. Based on the critique and tags, suggest 4 relevant architectural projects. For each project, provide:
                    - Project name and architect
                    - A detailed description of why it's relevant to the critique
                    - Key architectural features that relate to the critique
                    Format in JSON:
                    {
                        "projects": [
                            {
                                "title": "Project Name",
                                "architect": "Architect Name",
                                "description": "Detailed description of the project",
                                "relevance": "Why this project is relevant to the critique",
                                "key_features": ["Feature 1", "Feature 2", "Feature 3"]
                            }
                        ]
                    }
                    Make sure to always include all fields, especially key_features as an array.`
                },
                {
                    role: "user",
                    content: `Based on this architectural critique and tags, suggest 4 relevant architectural projects:\n\nCritique: ${critique}\n\nTags: ${JSON.stringify(tags)}`
                }
            ],
            temperature: 0.7,
            max_tokens: 1000
        });

        // Parse the response
        const responseText = referencesResponse.choices[0].message.content;
        let projectsData;
        try {
            projectsData = JSON.parse(responseText);
        } catch (error) {
            console.error('Error parsing references JSON:', error);
            return references;
        }

        // Convert to our reference format
        if (projectsData && projectsData.projects && Array.isArray(projectsData.projects)) {
            references.items = projectsData.projects.map(project => ({
                source: 'reference',
                title: project.title || 'Unknown Project',
                architect: project.architect || 'Unknown Architect',
                description: project.description || '',
                relevance: language === 'ko' ? 
                    `${project.title || 'Unknown Project'} - ${project.relevance || ''}` :
                    project.relevance || '',
                key_features: Array.isArray(project.key_features) ? project.key_features : []
            }));
        }

        return references;
    } catch (error) {
        console.error('Error in getReferences:', error);
        return references;
    }
}

app.listen(PORT, () => {
    console.log(`Server running at http://localhost:${PORT}`);
    console.log('Environment variables loaded:', { token: process.env.HUGGINGFACE_API_KEY ? 'Token exists' : 'No token found' });
});