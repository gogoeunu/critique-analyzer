function displayTags(tags) {
    if (!tags) {
        console.error('No tags provided to displayTags');
        return;
    }

    // Clear previous tags
    Object.keys(tags).forEach(category => {
        const container = document.getElementById(`${category}-tags`);
        if (container) {
            container.innerHTML = '';
        }
    });

    // Display new tags
    Object.entries(tags).forEach(([category, tagList]) => {
        if (!Array.isArray(tagList)) {
            console.error(`Tag list for ${category} is not an array:`, tagList);
            return;
        }

        const container = document.getElementById(`${category}-tags`);
        if (container && tagList.length > 0) {
            container.innerHTML = tagList.map(tag => 
                `<span class="tag">${tag}</span>`
            ).join('');
        }
    });
}

async function analyzeImage() {
    const fileInput = document.getElementById('imageInput');
    const file = fileInput.files[0];
    const language = document.getElementById('language').value;

    if (!file) {
        alert('Please select an image file');
        return;
    }

    const formData = new FormData();
    formData.append('image', file);
    formData.append('language', language);

    try {
        console.log('Sending image to backend...');
        const response = await fetch('http://localhost:3001/analyze', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Backend Error: ${response.status} - ${errorText}`);
        }

        const data = await response.json();
        console.log('Received response:', data);
        
        // Display tags
        if (data.tags) {
            displayTags(data.tags);
        } else {
            console.error('No tags in response:', data);
        }
        
        // Display critique and improvements
        const critiqueElement = document.getElementById('critique');
        const improvementsElement = document.getElementById('improvements');
        
        if (data.critique) {
            critiqueElement.innerHTML = `<h3>Critique Analysis</h3><p>${data.critique}</p>`;
        } else {
            critiqueElement.innerHTML = '<h3>Critique Analysis</h3><p>No critique available</p>';
        }
        
        if (data.improvements) {
            improvementsElement.innerHTML = `<h3>Improvement Plan</h3><p>${data.improvements}</p>`;
        } else {
            improvementsElement.innerHTML = '<h3>Improvement Plan</h3><p>No improvement suggestions available</p>';
        }

        // Show the results section
        document.getElementById('analysis-results').style.display = 'block';
    } catch (error) {
        console.error('Error in image analysis:', error);
        alert('Error analyzing image. Please try again.');
    }
} 