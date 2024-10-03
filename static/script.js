$(document).ready(function () {
    // Hide results section initially 
    $('#results').hide();

    // reset form on modal close 
    $('#modal').on('hidden.bs.modal', function () {
        $('#textForm').trigger('reset');
        $('#results').hide();
    });

    // Default operation is "suggest improvements"
    let operation = "suggest_improvements";
    $('#operation').val(operation);

    // Navbar operation selection
    $('#suggest-improvements-link').on('click', function () {
        operation = "suggest_improvements";
        $('#operation').val(operation);
        $('.navbar-nav .nav-link').removeClass('active');
        $(this).addClass('active');
    });

    $('#google-gemini-link').on('click', function () {
        operation = "google_gemini";
        $('#operation').val(operation);
        $('.navbar-nav .nav-link').removeClass('active');
        $(this).addClass('active');
    });

    // Handle form submission
    $('#textForm').on('submit', function (e) {
        e.preventDefault();

        // AJAX call to Flask endpoint
        $.ajax({
            type: 'POST',
            url: '/',
            data: new FormData(this),
            contentType: false,
            processData: false,
            success: function (response) {
                console.log("Response from server:", response);  // Debug log

                if (operation === 'suggest_improvements') {
                    displaySuggestImprovements(response);
                } else if (operation === 'google_gemini') {
                    displayGoogleGeminiResults(response);
                }
                $('#results').show();
            },
            error: function (error) {
                console.log("Error:", error);
                $('#results').html('<p class="text-danger">An error occurred while processing your request.</p>');
            }
        });
    });
});

// Function to display the "Suggest Improvements" results
function displaySuggestImprovements(data) {
    // Clear previous results
    $('#matchPercentage').html('');
    $('#missingKeywords').html('');
    $('#improvements').html('');
    $('#commonKeywords').html('');

    // Display match percentage with a progress bar
    if (data.match_percentage) {
        const matchPercentage = data.match_percentage;
        let progressBarHTML = `
            <div class="result-card">
                <h5><i class="fas fa-percentage card-icon"></i> Match Percentage</h5>
                <div class="progress">
                    <div class="progress-bar" role="progressbar" style="width: ${matchPercentage}%;" aria-valuenow="${matchPercentage}" aria-valuemin="0" aria-valuemax="100">
                        ${matchPercentage}%
                    </div>
                </div>
            </div>`;
        $('#matchPercentage').html(progressBarHTML);
    }

    // Display improvement suggestions
    if (data.improvements && data.improvements.length > 0) {
        let suggestionsHTML = '<div class="result-card"><h5><i class="fas fa-lightbulb card-icon"></i> Improvement Suggestions</h5><ul>';
        data.improvements.forEach(function (suggestion) {
            suggestionsHTML += `<li>${suggestion}</li>`;
        });
        suggestionsHTML += '</ul></div>';
        $('#improvements').html(suggestionsHTML);
    }

    // Display common keywords
    if (data.common_keywords) {
        let commonKeywordsHTML = '<div class="result-card"><h5><i class="fas fa-key card-icon"></i> Common Keywords</h5><ul>';
        for (const [keyword, count] of Object.entries(data.common_keywords)) {
            commonKeywordsHTML += `<li>${keyword} (${count})</li>`;
        }
        commonKeywordsHTML += '</ul></div>';
        $('#commonKeywords').html(commonKeywordsHTML);
    }

    // Show results section
    $('#results').show();
}

// Function to display the Google Gemini results
function displayGoogleGeminiResults(data) {
    // Clear previous results
    $('#matchPercentage').html('');
    $('#missingKeywords').html('');
    $('#candidateSummary').html('');
    $('#experience').html('');
    $('#missingSkills').html('');

    // Display job description match percentage with a progress bar
    if (data['Job Description Match']) {
        const matchPercentage = parseFloat(data['Job Description Match'].replace('%', ''));
        let progressBarHTML = `
            <div class="result-card">
                <h5><i class="fas fa-percentage card-icon"></i> Job Description Match</h5>
                <div class="progress">
                    <div class="progress-bar" role="progressbar" style="width: ${matchPercentage}%;" aria-valuenow="${matchPercentage}" aria-valuemin="0" aria-valuemax="100">
                        ${matchPercentage}%
                    </div>
                </div>
            </div>`;
        $('#matchPercentage').html(progressBarHTML);
    }

    // Display missing keywords if present
    if (data['Missing Keywords'] && data['Missing Keywords'].trim() !== '') {
        let missingKeywordsHTML = `
            <div class="result-card">
                <h5><i class="fas fa-exclamation-triangle card-icon"></i> Missing Keywords</h5>
                <p>${data['Missing Keywords']}</p>
            </div>`;
        $('#missingKeywords').html(missingKeywordsHTML);
    }

    // Display candidate summary
    if (data['Candidate Summary']) {
        let candidateSummaryHTML = `
            <div class="result-card">
                <h5><i class="fas fa-user card-icon"></i> Candidate Summary</h5>
                <p>${data['Candidate Summary']}</p>
            </div>`;
        $('#candidateSummary').html(candidateSummaryHTML);
    }

    // Display experience
    if (data['Experience']) {
        let experienceHTML = `
            <div class="result-card">
                <h5><i class="fas fa-briefcase card-icon"></i> Experience</h5>
                <p>${data['Experience']}</p>
            </div>`;
        $('#experience').html(experienceHTML);
    }

    // Display missing skills (Technical, Soft, Hard)
    let missingSkillsHTML = '';
    if (data['Missing Technical Skills'] && data['Missing Technical Skills'].trim() !== '') {
        missingSkillsHTML += `<li><strong>Technical Skills:</strong> ${data['Missing Technical Skills']}</li>`;
    }
    if (data['Missing Soft Skills'] && data['Missing Soft Skills'].trim() !== '') {
        missingSkillsHTML += `<li><strong>Soft Skills:</strong> ${data['Missing Soft Skills']}</li>`;
    }
    if (data['Missing Hard Skills'] && data['Missing Hard Skills'].trim() !== '') {
        missingSkillsHTML += `<li><strong>Hard Skills:</strong> ${data['Missing Hard Skills']}</li>`;
    }

    if (missingSkillsHTML) {
        $('#missingSkills').html(`<div class="result-card"><h5><i class="fas fa-tools card-icon"></i> Missing Skills</h5><ul>${missingSkillsHTML}</ul></div>`);
    }

    // Show results section
    $('#results').show();
}